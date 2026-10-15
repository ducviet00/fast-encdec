from collections import deque
from dataclasses import dataclass


@dataclass
class Block:
    ref_count: int = 0


class BlockManager:
    """Paged KV-cache block allocator with reference counting and copy-on-write."""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks = [Block() for _ in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))

    @property
    def num_free_blocks(self) -> int:
        """Return the number of currently unallocated KV blocks."""
        return len(self.free_block_ids)

    def _allocate_block(self) -> int:
        """Pop a free block ID and initialize its reference count to 1."""
        if not self.free_block_ids:
            raise RuntimeError("KV cache is full; increase num_blocks")
        block_id = self.free_block_ids.popleft()
        self.blocks[block_id].ref_count = 1
        return block_id

    def _deallocate_block(self, block_id: int) -> None:
        """Return a block with zero references back to the free pool."""
        self.blocks[block_id].ref_count = 0
        self.free_block_ids.append(block_id)

    def required_new_blocks(self, seq, num_new_tokens: int) -> int:
        """Return an upper bound on fresh blocks needed (growth + copy-on-write)."""
        if num_new_tokens <= 0:
            return 0
        bs = self.block_size
        start, end = seq.num_cached_tokens, seq.num_cached_tokens + num_new_tokens
        old = len(seq.block_table)
        new_blocks = max(0, (end + bs - 1) // bs - old)
        cow_blocks = sum(
            i < old and self.blocks[seq.block_table[i]].ref_count > 1
            for i in range(start // bs, (end - 1) // bs + 1)
        )
        return new_blocks + cow_blocks

    def ensure_capacity(self, seq, num_new_tokens: int, copy_block) -> None:
        """Allocate new blocks for seq and copy-on-write any shared blocks to be written."""
        bs = self.block_size
        start, end = seq.num_cached_tokens, seq.num_cached_tokens + num_new_tokens
        needed = (end + bs - 1) // bs
        while len(seq.block_table) < needed:
            seq.block_table.append(self._allocate_block())
        for i in range(start // bs, (end - 1) // bs + 1):
            block_id = seq.block_table[i]
            if self.blocks[block_id].ref_count > 1:
                new_block_id = self._allocate_block()
                copy_block(block_id, new_block_id)
                self.blocks[block_id].ref_count -= 1
                seq.block_table[i] = new_block_id

    def fork(self, parent, child) -> None:
        """Share parent's block table with child by incrementing block refcounts."""
        child.block_table = list(parent.block_table)
        for block_id in child.block_table:
            self.blocks[block_id].ref_count += 1

    def deallocate(self, seq) -> None:
        """Decrement refcounts for all blocks in seq and free any that reach zero."""
        for block_id in reversed(seq.block_table):
            self.blocks[block_id].ref_count -= 1
            if self.blocks[block_id].ref_count == 0:
                self._deallocate_block(block_id)
        seq.block_table.clear()
