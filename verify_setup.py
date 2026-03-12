"""
Verification script to check if all dependencies are installed correctly
"""

import sys

def check_import(module_name, package_name=None):
    """Try to import a module and report status"""
    if package_name is None:
        package_name = module_name
    
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name:20s} - v{version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name:20s} - NOT INSTALLED")
        print(f"   Error: {e}")
        return False

def main():
    print("=" * 60)
    print("  Driver Drowsiness Detection - Setup Verification")
    print("=" * 60)
    print()
    
    print("Python Version:")
    print(f"  {sys.version}")
    print()
    
    print("Checking required packages...")
    print("-" * 60)
    
    results = {}
    
    # Core packages
    print("\n[Core Deep Learning]")
    results['tensorflow'] = check_import('tensorflow')
    results['keras'] = check_import('keras')
    
    # Computer Vision
    print("\n[Computer Vision]")
    results['cv2'] = check_import('cv2', 'opencv-python')
    results['mediapipe'] = check_import('mediapipe')
    
    # Numerical
    print("\n[Numerical Computing]")
    results['numpy'] = check_import('numpy')
    results['PIL'] = check_import('PIL', 'pillow')
    
    # Data Science
    print("\n[Data Science & ML]")
    results['matplotlib'] = check_import('matplotlib')
    results['seaborn'] = check_import('seaborn')
    results['sklearn'] = check_import('sklearn', 'scikit-learn')
    results['plotly'] = check_import('plotly')
    results['pandas'] = check_import('pandas')
    
    # Utilities
    print("\n[Utilities]")
    results['tqdm'] = check_import('tqdm')
    results['scipy'] = check_import('scipy')
    
    # Optional packages
    print("\n[Optional Packages]")
    check_import('torch', 'pytorch (optional)')
    check_import('onnx', 'onnx (optional)')
    check_import('flask', 'flask (optional)')
    
    # Summary
    print()
    print("=" * 60)
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    print(f"Summary: {passed}/{total} required packages installed")
    
    if failed > 0:
        print(f"\n⚠️  {failed} package(s) missing!")
        print("Please run: pip install -r requirements.txt")
        print()
        return 1
    else:
        print("\n✅ All required packages are installed!")
        print()
        
        # Additional checks
        print("Performing additional checks...")
        print("-" * 60)
        
        # Check TensorFlow GPU
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"✅ TensorFlow GPU available: {len(gpus)} GPU(s) detected")
                for gpu in gpus:
                    print(f"   - {gpu.name}")
            else:
                print("ℹ️  TensorFlow will use CPU (GPU not detected)")
        except Exception as e:
            print(f"⚠️  Error checking TensorFlow GPU: {e}")
        
        # Check model files
        import os
        print("\nChecking model files...")
        model_files = [
            'B0_16_batches.weights.keras',
            'B1_16_batches.weights.keras',
            'blaze_face_short_range.tflite'
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                size_mb = os.path.getsize(model_file) / (1024 * 1024)
                print(f"✅ {model_file:35s} ({size_mb:.1f} MB)")
            else:
                print(f"❌ {model_file:35s} - NOT FOUND")
        
        # Check webcam
        print("\nChecking webcam...")
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    print(f"✅ Webcam accessible (resolution: {w}x{h})")
                else:
                    print("⚠️  Webcam opened but cannot read frames")
                cap.release()
            else:
                print("❌ Cannot open webcam")
        except Exception as e:
            print(f"⚠️  Error checking webcam: {e}")
        
        print()
        print("=" * 60)
        print("🎉 Setup verification complete!")
        print()
        print("You can now run:")
        print("  python main.py")
        print()
        return 0

if __name__ == "__main__":
    sys.exit(main())
