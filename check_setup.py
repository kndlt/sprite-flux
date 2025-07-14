#!/usr/bin/env python3
"""
Verify that all dependencies are available for LoRA fine-tuning
"""

def check_imports():
    """Check if all required packages can be imported"""
    checks = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("accelerate", "Accelerate"),
        ("peft", "PEFT"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("tqdm", "tqdm"),
        ("safetensors", "SafeTensors"),
        ("datasets", "Datasets"),
        ("bitsandbytes", "BitsAndBytes (optional)"),
    ]
    
    print("Checking dependencies for LoRA fine-tuning...")
    print("=" * 50)
    
    all_good = True
    for module, name in checks:
        try:
            if module == "bitsandbytes":
                # Special handling for bitsandbytes due to compilation issues
                import importlib
                importlib.import_module(module)
            else:
                __import__(module)
            print(f"✅ {name:<20} - Available")
        except ImportError as e:
            if "bitsandbytes" in module:
                print(f"⚠️  {name:<20} - Optional, compilation issue (fine for basic training)")
            else:
                print(f"❌ {name:<20} - Missing: {e}")
                all_good = False
        except Exception as e:
            if "bitsandbytes" in module:
                print(f"⚠️  {name:<20} - Optional, compilation issue (fine for basic training)")
            else:
                print(f"❌ {name:<20} - Error: {e}")
                all_good = False
    
    print("=" * 50)
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA Available - {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
        else:
            print("❌ CUDA not available - training will be very slow")
            all_good = False
    except:
        print("❌ Could not check CUDA availability")
        all_good = False
    
    print("=" * 50)
    
    # Check if input directory exists
    import os
    input_dir = "inputs/lora-fine-tuning"
    if os.path.exists(input_dir):
        image_count = len([f for f in os.listdir(input_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        print(f"✅ Input directory exists: {input_dir}")
        print(f"   Found {image_count} images")
        if image_count == 0:
            print("⚠️  No training images found - add images to start training")
    else:
        print(f"❌ Input directory missing: {input_dir}")
        print("   Create this directory and add your training images")
        all_good = False
    
    print("=" * 50)
    
    if all_good:
        print("🎉 All dependencies are ready for LoRA fine-tuning!")
        print("   Add your training images to inputs/lora-fine-tuning/ to get started")
    else:
        print("❗ Some issues found - please resolve them before training")
    
    return all_good

if __name__ == "__main__":
    check_imports()
