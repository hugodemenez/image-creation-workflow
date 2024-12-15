import os
from datetime import datetime
from typing import Optional
from PIL import Image
import torch
from diffusers import DiffusionPipeline
from dotenv import load_dotenv
import argparse


class ImageGenerator:
    def __init__(
        self, 
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        lora_id: Optional[str] = None
    ):
        """
        Initialize the image generator

        Args:
            model_id (str): Base model ID to use for generation
            lora_id (str, optional): LoRA model ID to load
        """
        # Load environment variables
        load_dotenv()
        
        # Get token from environment
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise ValueError("Please set HUGGINGFACE_TOKEN in your .env file")

        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_auth_token=token  # Add authentication token here
        )
        
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
            
        if lora_id:
            self.pipe.load_lora_weights(lora_id, use_auth_token=token)  # Add token here too

    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 1024,
        height: int = 512,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate an image from a text prompt

        Args:
            prompt (str): Text prompt to generate image from
            negative_prompt (str, optional): What not to include in the image
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): How closely to follow the prompt
            width (int): Image width in pixels
            height (int): Image height in pixels
            seed (int, optional): Random seed for reproducibility

        Returns:
            PIL.Image: Generated image
        """
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None

        try:
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
            )
            
            return output.images[0]  # Return the first generated image

        except Exception as e:
            print(f"Error generating image: {str(e)}")
            raise

    def save_image(self, image: Image.Image, output_dir: str = "generated_images"):
        """
        Save the generated image with a timestamp

        Args:
            image (PIL.Image): Image to save
            output_dir (str): Directory to save the image in
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        print(f"Image saved to {filepath}")


def create_blended_prompt(user_prompt: str) -> str:
    """
    Blend user prompt with the base style prompt

    Args:
        user_prompt (str): User's input prompt

    Returns:
        str: Blended prompt with style elements
    """
    style_elements = (
        "minimalist, impressionism, negative space"
    )

    return f"{user_prompt}, {style_elements}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate an AI image using FLUX.1 model"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="a monochromatic pencil sketch of a ragdoll cat, blue eyes",
        help="The main subject/scene to generate",
    )
    args = parser.parse_args()

    # Initialize generator
    generator = ImageGenerator()

    # Create the final prompt
    final_prompt = create_blended_prompt(args.prompt)
    print(f"Generated prompt: {final_prompt}")

    try:
        # Generate image
        image = generator.generate_image(
            prompt=final_prompt,
            width=512,
            height=512,
        )

        # Save the image
        generator.save_image(image)

    except Exception as e:
        print(f"Failed to generate image: {str(e)}")


if __name__ == "__main__":
    main()
