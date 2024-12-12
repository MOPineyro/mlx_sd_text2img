import mlx.core as mx
import mlx.nn as nn
from stable_diffusion import StableDiffusion
from PIL import Image
import numpy as np
from tqdm import tqdm
import gradio as gr

# Initialize model
model = StableDiffusion(
    "stabilityai/stable-diffusion-2-1-base",
    float16=True
)

# Optional: Quantize model for better performance
nn.quantize(model.text_encoder, class_predicate=lambda _, m: isinstance(m, nn.Linear))
nn.quantize(model.unet, group_size=32, bits=8)

# Ensure models are loaded
model.ensure_models_are_loaded()

def generate_image(
    prompt,
    negative_prompt,
    num_steps,
    guidance_scale,
    width,
    height,
    seed
):
    """Generate image from prompt using Stable Diffusion"""
    try:
        # Generate latents
        latents = model.generate_latents(
            prompt,
            n_images=1,
            num_steps=num_steps,
            cfg_weight=guidance_scale,
            negative_text=negative_prompt,
            seed=seed if seed != 0 else None,
            latent_size=(height//8, width//8)
        )
        
        # Process latents
        for x_t in tqdm(latents, total=num_steps):
            mx.eval(x_t)
        
        # Decode to image
        decoded = model.decode(x_t)
        mx.eval(decoded)
        
        # Convert to PIL
        image_array = np.array(decoded[0] * 255).astype(np.uint8)
        image = Image.fromarray(image_array)
        
        return image, "¡Generación exitosa! / Generation successful!"
    except Exception as e:
        return None, f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="MLX Text to Image Generator") as demo:
    gr.Markdown(
        """
        # 🎨 MLX Text to Image Generator
        Generate images from text descriptions using Stable Diffusion optimized for Apple Silicon.
        """
    )
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the image you want to generate...",
                lines=3
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="Describe what you DON'T want in the image...",
                lines=2,
                value="blurry, bad quality, distorted"
            )
            
            with gr.Row():
                width = gr.Slider(minimum=256, maximum=1024, step=64, value=512, label="Width")
                height = gr.Slider(minimum=256, maximum=1024, step=64, value=512, label="Height")
            
            with gr.Row():
                steps = gr.Slider(minimum=1, maximum=50, step=1, value=20, label="Steps")
                guidance = gr.Slider(minimum=1, maximum=20, step=0.5, value=7.5, label="Guidance Scale")
                seed = gr.Number(value=0, label="Seed (0 for random)", precision=0)
            
            generate_btn = gr.Button("🎨 Generate Image", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="Generated Image", type="pil")
            output_text = gr.Textbox(label="Status", interactive=False)
    
    # Connect the function
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, steps, guidance, width, height, seed],
        outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True) 