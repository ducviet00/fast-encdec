"""Paged KV-cache block allocator with reference counting.

Blocks are shared between beam hypotheses; a block is copied only when a
sequence is about to write into a shared block (copy-on-write).  This mirrors
vLLM's beam-search behaviour while staying tiny.
"""


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.ref_count = [0] * num_blocks

    def _allocate(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("KV cache is full; increase num_blocks")
        block = self.free_blocks.pop()
        self.ref_count[block] = 1
        return block

    def required_new_blocks(self, seq, num_new_tokens: int) -> int:
        """Upper bound on the fresh blocks :meth:`ensure_capacity` allocates.

        Counts the blocks the sequence still has to grow into plus one
        copy-on-write copy per existing block it is about to write while
        shared.  It can overcount when several sequences share a block that
        only the first of them copies, so it is safe to reserve with.
        """
        if num_new_tokens <= 0:
            return 0
        bs = self.block_size
        start = seq.num_cached_tokens
        end = start + num_new_tokens
        old = len(seq.block_table)
        count = max(0, (end + bs - 1) // bs - old)
        for i in range(start // bs, (end - 1) // bs + 1):
            if i < old and self.ref_count[seq.block_table[i]] > 1:
                count += 1
        return count

    def ensure_capacity(self, seq, num_new_tokens: int, copy_block) -> None:
        """Grow ``seq`` and make every block it will write to private."""
        bs = self.block_size
        start = seq.num_cached_tokens
        end = start + num_new_tokens
        needed = (end + bs - 1) // bs
        while len(seq.block_table) < needed:
            seq.block_table.append(self._allocate())
        for i in range(start // bs, (end - 1) // bs + 1):
            block = seq.block_table[i]
            if self.ref_count[block] > 1:  # shared -> copy before writing
                new_block = self._allocate()
                copy_block(block, new_block)
                self.ref_count[block] -= 1
                seq.block_table[i] = new_block

    def fork(self, parent, child) -> None:
        """Share ``parent``'s blocks with ``child`` (no data copy)."""
        child.block_table = list(parent.block_table)
        for block in child.block_table:
            self.ref_count[block] += 1

    def free(self, seq) -> None:
        for block in seq.block_table:
            self.ref_count[block] -= 1
            if self.ref_count[block] == 0:
                self.free_blocks.append(block)
        seq.block_table = []
