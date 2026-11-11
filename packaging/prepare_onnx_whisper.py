"""Keep Whisper FP16 weights on disk, execute portable ONNX operators in FP32.

Conversion tooling is build-only. No onnx/transformers/converter is shipped.
Input files are preserved; the output directory must be new.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import copy


def adapt_graph(graph, onnx):
    from onnx import helper, numpy_helper, TensorProto
    casts = []
    for tensor in graph.initializer:
        if tensor.data_type == TensorProto.FLOAT16:
            original = tensor.name
            tensor.name = original + "__vbg_fp16_storage"
            casts.append(helper.make_node("Cast", [tensor.name], [original], to=TensorProto.FLOAT,
                                          name=original + "__vbg_fp32_compute"))
    for value in [*graph.input, *graph.output, *graph.value_info]:
        if value.type.HasField("tensor_type") and value.type.tensor_type.elem_type == TensorProto.FLOAT16:
            value.type.tensor_type.elem_type = TensorProto.FLOAT
    for node in graph.node:
        for attr in node.attribute:
            if node.op_type == "Cast" and attr.name == "to" and attr.i == TensorProto.FLOAT16:
                attr.i = TensorProto.FLOAT
            if attr.type == onnx.AttributeProto.GRAPH:
                adapt_graph(attr.g, onnx)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for child in attr.graphs:
                    adapt_graph(child, onnx)
            elif attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == TensorProto.FLOAT16:
                attr.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(attr.t).astype("float32"), attr.t.name))
    nodes = [*casts, *graph.node]
    del graph.node[:]
    graph.node.extend(nodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory; source models are never modified")
    import onnx
    args.output.mkdir(parents=True)
    records = {}
    for path in sorted(args.source.rglob("*")):
        relative = path.relative_to(args.source)
        if not path.is_file() or path.suffix not in {".onnx", ".json"} or path.name.endswith(".ranges.json"):
            continue
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".onnx":
            model = onnx.load(path)
            adapt_graph(model.graph, onnx)
            if 'decoder_model_merged' in path.name:
                # DirectML can return invalid values for the merged decoder's
                # empty encoder-cache outputs. Separate the two branches and
                # share one external weight file, avoiding If/empty outputs.
                branch = next(n for n in model.graph.node if n.op_type == 'If')
                first = True
                for attribute in branch.attribute:
                    child = copy.deepcopy(attribute.g)
                    variant = copy.deepcopy(model)
                    graph = variant.graph
                    names = {o.name:name for o,name in zip(child.output,branch.output)}
                    for node in child.node:
                        for i,name in enumerate(node.input):
                            node.input[i] = names.get(name,name)
                        for i,name in enumerate(node.output):
                            node.output[i] = names.get(name,name)
                    nodes = []
                    for node in model.graph.node:
                        nodes.extend(child.node if node.op_type == 'If' else [node])
                    del graph.node[:]
                    graph.node.extend(nodes)
                    graph.initializer.extend(child.initializer)
                    outputs = [o for o in graph.output if not (attribute.name == 'then_branch' and '.encoder.' in o.name)]
                    del graph.output[:]
                    graph.output.extend(outputs)
                    used = {name for n in graph.node for name in n.input}
                    inputs = [v for v in graph.input if v.name in used]
                    del graph.input[:]
                    graph.input.extend(inputs)
                    del graph.value_info[:]
                    name = 'decoder_with_past_model.onnx' if attribute.name == 'then_branch' else 'decoder_model.onnx'
                    # Both branches contain the same root weights. Store those
                    # once; branch-local constants stay embedded in the header.
                    if first:
                        onnx.save_model(variant, args.output/'onnx'/name, save_as_external_data=True,
                                        all_tensors_to_one_file=True, location='decoder.weights',size_threshold=1024)
                        shared = {t.name:copy.deepcopy(t) for t in variant.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL}
                        first = False
                    else:
                        for tensor in graph.initializer:
                            if tensor.name in shared:
                                tensor.CopyFrom(shared[tensor.name])
                        onnx.save(variant, args.output/'onnx'/name)
                    onnx.checker.check_model(str(args.output/'onnx'/name))
                    print(name,flush=True)
                records[relative.as_posix()] = {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "adapted_to": ["onnx/decoder_model.onnx", "onnx/decoder_with_past_model.onnx", "onnx/decoder.weights"]}
                continue
            else:
                onnx.checker.check_model(model)
                onnx.save(model, target)
        else:
            shutil.copy2(path, target)
        records[relative.as_posix()] = {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                      "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        print(relative, target.stat().st_size, flush=True)
    (args.output / "voicebankgen-onnx.json").write_text(json.dumps({"compute_type":"fp32",
        "transformation":"split-decoder-fp32-compute-fp16-storage-v1", "upstream":"onnx-community/whisper-small",
        "revision":"36050c46d777d46dc4b5f43f6d90574fc38f8732", "files":records},indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
