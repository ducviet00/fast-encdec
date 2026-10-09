"""Paged KV-cache block allocator with reference counting.

Blocks are shared between beam hypotheses; a block is copied only when a
sequence is about to write into a shared block (copy-on-write).  This mirrors
vLLM's beam-search behaviour while staying tiny.
"""

from collections import deque


class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def _allocate_block(self) -> int:
        if not self.free_block_ids:
            raise RuntimeError("KV cache is full; increase num_blocks")
        block_id = self.free_block_ids.popleft()
        self.blocks[block_id].ref_count = 1
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int) -> None:
        self.blocks[block_id].ref_count = 0
        self.used_block_ids.discard(block_id)
        self.free_block_ids.append(block_id)

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
            if i < old and self.blocks[seq.block_table[i]].ref_count > 1:
                count += 1
        return count

    def ensure_capacity(self, seq, num_new_tokens: int, copy_block) -> None:
        """Grow ``seq`` and make every block it will write to private."""
        bs = self.block_size
        start = seq.num_cached_tokens
        end = start + num_new_tokens
        needed = (end + bs - 1) // bs
        while len(seq.block_table) < needed:
            seq.block_table.append(self._allocate_block())
        for i in range(start // bs, (end - 1) // bs + 1):
            block_id = seq.block_table[i]
            if self.blocks[block_id].ref_count > 1:  # shared -> copy before writing
                new_block_id = self._allocate_block()
                copy_block(block_id, new_block_id)
                self.blocks[block_id].ref_count -= 1
                seq.block_table[i] = new_block_id

    def fork(self, parent, child) -> None:
        """Share ``parent``'s blocks with ``child`` (no data copy)."""
        child.block_table = list(parent.block_table)
        for block_id in child.block_table:
            self.blocks[block_id].ref_count += 1

    def deallocate(self, seq) -> None:
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.block_table = []
