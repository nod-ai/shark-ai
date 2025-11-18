from iree.turbine.aot import (
    ExportOutput,
    FxProgramsBuilder,
    export,
    externalize_module_parameters,
    save_module_parameters,
    decompositions,
)
import os


def export_sdxl_model(
    hf_model_name,
    component,
    batch_size,
    height,
    width,
    precision="fp16",
    max_length=64,
    external_weights=None,
    external_weights_file=None,
    decomp_attn=False,
    quant_path=None,
    scheduler_config_path=None,
    weights_only=False,
) -> ExportOutput:
    import torch

    def check_torch_version(begin: tuple, end: tuple):
        pass

    decomp_list = [torch.ops.aten.logspace]
    if decomp_attn == True:
        decomp_list = [
            torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops.aten.scaled_dot_product_attention.default,
            torch.ops.aten.scaled_dot_product_attention,
        ]
    with decompositions.extend_aot_decompositions(
        from_current=True,
        add_ops=decomp_list,
    ):
        if component == "clip":
            from amdsharktank.torch_exports.sdxl.clip import get_clip_model_and_inputs

            module_name = "compiled_clip"
            model, sample_clip_inputs = get_clip_model_and_inputs(
                hf_model_name, max_length, precision, batch_size
            )
            if external_weights:
                # Transformers (model source) registers position ids as non-persistent.
                # This causes externalization to think it's a user input, and since it's not,
                # we end up trying to do ops on a !torch.None instead of a tensor.
                for buffer_name, buffer in model.named_buffers(recurse=True):
                    mod_name_list = buffer_name.split(".")
                    buffer_id = mod_name_list.pop()
                    parent = model
                    for i in mod_name_list:
                        parent = getattr(parent, i)
                    parent.register_buffer(buffer_id, buffer, persistent=True)
            model.to("cpu")
            fxb = FxProgramsBuilder(model)

            @fxb.export_program(
                args=(sample_clip_inputs,),
            )
            def encode_prompts(
                module,
                inputs,
            ):
                return module.forward(**inputs)

        elif component in ["unet", "punet", "scheduled_unet"]:
            check_torch_version((2, 4, 1), (2, 6, 0))
            from amdsharktank.torch_exports.sdxl.unet import (
                get_scheduled_unet_model_and_inputs,
                get_punet_model_and_inputs,
            )

            if component in ["unet", "punet"]:
                module_name = "compiled_punet"
                implementation = get_punet_model_and_inputs
            else:
                module_name = "compiled_spunet"
                implementation = get_scheduled_unet_model_and_inputs
            (model, sample_init_inputs, sample_forward_inputs,) = implementation(
                hf_model_name,
                height,
                width,
                max_length,
                precision,
                batch_size,
                external_weights_file,
                quant_path,
                scheduler_config_path,
            )
            if external_weights:
                externalize_module_parameters(model.cond_model)
            if component == "scheduled_unet":
                fxb = FxProgramsBuilder(model)

                @fxb.export_program(
                    args=(sample_init_inputs,),
                )
                def run_initialize(
                    module,
                    inputs,
                ):
                    return module.initialize(*inputs)

                @fxb.export_program(
                    args=(sample_forward_inputs,),
                )
                def run_forward(
                    module,
                    inputs,
                ):
                    return module.forward(*inputs)

                return export(fxb, module_name=module_name)
            else:
                return export(
                    model, kwargs=sample_forward_inputs, module_name="compiled_punet"
                )
        elif component == "scheduler":
            module_name = "compiled_scheduler"
            from amdsharktank.torch_exports.sdxl.scheduler import (
                get_scheduler_model_and_inputs,
            )

            model, init_args, prep_args, step_args = get_scheduler_model_and_inputs(
                hf_model_name if not scheduler_config_path else scheduler_config_path,
                batch_size,
                height,
                width,
                precision,
            )
            fxb = FxProgramsBuilder(model)

            @fxb.export_program(
                args=(init_args,),
            )
            def run_initialize(module, sample):
                return module.initialize(*sample)

            @fxb.export_program(
                args=(prep_args,),
            )
            def run_scale(module, inputs):
                return module.scale_model_input(*inputs)

            @fxb.export_program(
                args=(step_args,),
            )
            def run_step(module, inputs):
                return module.step(*inputs)

        elif component == "vae":
            from amdsharktank.torch_exports.sdxl.vae import get_vae_model_and_inputs

            module_name = "compiled_vae"
            if quant_path and os.path.exists(
                os.path.join(quant_path, "vae.safetensors")
            ):
                vae_path = os.path.join(quant_path, "vae.safetensors")
            else:
                vae_path = None
            model, encode_args, decode_args = get_vae_model_and_inputs(
                hf_model_name,
                height,
                width,
                precision=precision,
                batch_size=batch_size,
                custom_vae_path=vae_path,
            )
            fxb = FxProgramsBuilder(model)

            @fxb.export_program(
                args=(decode_args,),
            )
            def decode(
                module,
                inputs,
            ):
                return module.decode(*inputs)

        else:
            raise ValueError("Unimplemented: ", component)

    if external_weights:
        externalize_module_parameters(model)
    if external_weights_file:
        save_module_parameters(external_weights_file, model)
    module = export(fxb, module_name=module_name)
    return module
