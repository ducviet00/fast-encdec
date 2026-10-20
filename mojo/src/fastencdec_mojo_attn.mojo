"""Python extension module for the fast-encdec Mojo paged-attention kernel.

Everything crossing the boundary is a raw address / integer, so a call costs a
handful of `PyLong` conversions and no tensor bookkeeping. The Python side
(``fastencdec/layers/mojo_attn.py``) owns the tensors and passes
``tensor.data_ptr()``.

Build with:  pixi run -m ../pixi.toml build     (see ../pixi.toml)
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.runtime import initialize_runtime

from attention import Job, run


@export
def PyInit_fastencdec_mojo_attn() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("fastencdec_mojo_attn")
        m.def_function[attention]("attention")
        return m.finalize()
    except e:
        abort(String("error creating fastencdec_mojo_attn: ", e))


def attention(
    q: PythonObject,
    k: PythonObject,
    v: PythonObject,
    k_cache: PythonObject,
    v_cache: PythonObject,
    dst: PythonObject,
    slots: PythonObject,
    block_table: PythonObject,
    lens: PythonObject,
    num_seqs: PythonObject,
    num_heads: PythonObject,
    num_kv_heads: PythonObject,
    head_dim: PythonObject,
    block_size: PythonObject,
    bt_stride: PythonObject,
    q_stride: PythonObject,
    kv_stride: PythonObject,
    dst_stride: PythonObject,
    dtype: PythonObject,
    scale: PythonObject,
    num_workers: PythonObject,
    do_store: PythonObject,
    do_attend: PythonObject,
) raises:
    """Run the paged attention kernel (scatter K/V when ``do_store``).

    ``dtype`` is 0 for float32 and 1 for bfloat16;
    ``head_dim`` must be 32, 64 or 128.
    """
    initialize_runtime()
    var job = Job()
    job.q = Int(py=q)
    job.k = Int(py=k)
    job.v = Int(py=v)
    job.k_cache = Int(py=k_cache)
    job.v_cache = Int(py=v_cache)
    job.dst = Int(py=dst)
    job.slots = Int(py=slots)
    job.block_table = Int(py=block_table)
    job.lens = Int(py=lens)
    job.num_seqs = Int(py=num_seqs)
    job.num_heads = Int(py=num_heads)
    job.num_kv_heads = Int(py=num_kv_heads)
    job.head_dim = Int(py=head_dim)
    job.block_size = Int(py=block_size)
    job.bt_stride = Int(py=bt_stride)
    job.q_stride = Int(py=q_stride)
    job.kv_stride = Int(py=kv_stride)
    job.dst_stride = Int(py=dst_stride)
    job.scale = Float32(Float64(py=scale))
    run(
        job,
        dtype_code=Int(py=dtype),
        num_workers=Int(py=num_workers),
        do_store=Bool(do_store),
        do_attend=Bool(do_attend),
    )
