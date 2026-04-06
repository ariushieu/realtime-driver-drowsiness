"""
Benchmark Script for Driver Drowsiness Detection System
Đánh giá hiệu năng mô hình B0 và B1 trên thiết bị thực tế
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import os
from datetime import datetime
from collections import deque

# Import detector
import main as driver_detection


class BenchmarkRunner:
    def __init__(self, model_path, num_frames=300):
        """
        Args:
            model_path: Path to model weights
            num_frames: Number of frames to benchmark
        """
        self.model_path = model_path
        self.num_frames = num_frames
        self.results = {
            'model_path': model_path,
            'variant': 'B1' if 'B1' in model_path else 'B0',
            'num_frames': num_frames,
            'inference_times': [],
            'fps_values': [],
            'face_detection_times': [],
            'preprocessing_times': [],
            'facemesh_times': [],
            'fusion_times': [],
            'total_times': [],
        }
    
    def run(self):
        """Run benchmark"""
        print("=" * 70)
        print(f"🔬 BENCHMARK: {self.results['variant']}")
        print(f"   Model: {self.model_path}")
        print(f"   Frames: {self.num_frames}")
        print("=" * 70)
        
        # Initialize detector
        print("\n📦 Loading model...")
        detector = driver_detection.ImprovedDriverDetector(
            model_path=self.model_path,
            use_mediapipe=True,
            use_facemesh=True
        )
        
        # Open webcam
        print("📹 Opening webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return None
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"\n🚀 Starting benchmark ({self.num_frames} frames)...")
        print("   (Nhìn thẳng vào camera để có kết quả tốt nhất)\n")
        
        frame_count = 0
        faces_detected = 0
        
        try:
            while frame_count < self.num_frames:
                loop_start = time.time()

                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)

                # Face detection timing
                face_start = time.time()
                if detector.use_mediapipe:
                    faces = detector.detect_faces_mediapipe(frame)
                else:
                    faces = detector.detect_faces_haar(frame)
                face_time = time.time() - face_start
                self.results['face_detection_times'].append(face_time)

                # FaceMesh landmark timing
                facial_metrics = None
                if detector.use_facemesh and len(faces) > 0:
                    mesh_start = time.time()
                    face_landmarks = detector.detect_face_landmarks(frame)
                    if face_landmarks:
                        facial_metrics = detector.extract_facial_metrics(frame, face_landmarks)
                        detector.draw_facial_features(frame, facial_metrics)
                    mesh_time = time.time() - mesh_start
                    self.results['facemesh_times'].append(mesh_time)

                predictions_data = []

                if len(faces) > 0:
                    faces_detected += 1
                    x, y, w, h = faces[0]

                    if w >= 50 and h >= 50:
                        face_roi = frame[y:y+h, x:x+w]

                        # Preprocessing timing
                        prep_start = time.time()
                        face_input = detector.preprocess_face(face_roi)
                        prep_time = time.time() - prep_start
                        self.results['preprocessing_times'].append(prep_time)

                        # Inference timing
                        inf_start = time.time()
                        predictions = detector.model.predict(face_input, verbose=0)
                        inf_time = time.time() - inf_start
                        self.results['inference_times'].append(inf_time)

                        # Full pipeline: smooth -> fuse -> stabilize
                        fusion_start = time.time()
                        smoothed = detector.smooth_predictions(predictions[0])
                        predicted_class = np.argmax(smoothed)
                        confidence = float(smoothed[predicted_class])
                        predicted_class, confidence = detector.fuse_prediction(
                            predicted_class, confidence, facial_metrics
                        )
                        predicted_class, confidence = detector.stabilize_class(
                            predicted_class, confidence
                        )
                        fusion_time = time.time() - fusion_start
                        self.results['fusion_times'].append(fusion_time)

                        predictions_data.append((predicted_class, confidence, smoothed))

                # Draw dashboard UI (same as main.py demo)
                detector.draw_ui(frame, faces, predictions_data, facial_metrics,
                                 frame_count)

                # Benchmark progress overlay
                progress = frame_count / self.num_frames
                bar_w = 200
                bar_x = (frame.shape[1] - bar_w) // 2
                bar_y = frame.shape[0] - 60
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12),
                              (50, 50, 50), -1)
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + int(bar_w * progress), bar_y + 12),
                              (0, 200, 200), -1)
                cv2.putText(frame, f"Benchmark: {frame_count}/{self.num_frames}",
                            (bar_x, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 200, 200), 1, cv2.LINE_AA)
                # Total frame time
                loop_time = time.time() - loop_start
                self.results['total_times'].append(loop_time)
                fps = 1.0 / loop_time if loop_time > 0 else 0
                self.results['fps_values'].append(fps)

                frame_count += 1

                # Progress indicator (terminal)
                if frame_count % 50 == 0:
                    avg_fps = np.mean(self.results['fps_values'][-50:])
                    print(f"   Frame {frame_count}/{self.num_frames} | FPS: {avg_fps:.1f}")

                cv2.imshow('AI Driver Monitoring - Benchmark', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⏹️  Benchmark interrupted by user")
                    break
        
        except KeyboardInterrupt:
            print("\n⏹️  Benchmark interrupted")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Cleanup
            if hasattr(detector, 'face_detector_task'):
                detector.face_detector_task.close()
            if hasattr(detector, 'face_landmarker_task'):
                detector.face_landmarker_task.close()
        
        # Calculate statistics
        self.results['frames_processed'] = frame_count
        self.results['faces_detected'] = faces_detected
        self.results['face_detection_rate'] = faces_detected / frame_count if frame_count > 0 else 0
        
        if self.results['inference_times']:
            self.results['avg_inference_ms'] = np.mean(self.results['inference_times']) * 1000
            self.results['std_inference_ms'] = np.std(self.results['inference_times']) * 1000
            self.results['min_inference_ms'] = np.min(self.results['inference_times']) * 1000
            self.results['max_inference_ms'] = np.max(self.results['inference_times']) * 1000
        
        if self.results['fps_values']:
            self.results['avg_fps'] = np.mean(self.results['fps_values'])
            self.results['std_fps'] = np.std(self.results['fps_values'])
            self.results['min_fps'] = np.min(self.results['fps_values'])
            self.results['max_fps'] = np.max(self.results['fps_values'])
        
        if self.results['face_detection_times']:
            self.results['avg_face_detection_ms'] = np.mean(self.results['face_detection_times']) * 1000
        
        if self.results['preprocessing_times']:
            self.results['avg_preprocessing_ms'] = np.mean(self.results['preprocessing_times']) * 1000

        if self.results['facemesh_times']:
            self.results['avg_facemesh_ms'] = np.mean(self.results['facemesh_times']) * 1000

        if self.results['fusion_times']:
            self.results['avg_fusion_ms'] = np.mean(self.results['fusion_times']) * 1000

        if self.results['total_times']:
            self.results['avg_total_ms'] = np.mean(self.results['total_times']) * 1000
        
        return self.results
    
    def print_results(self):
        """Print benchmark results"""
        r = self.results
        
        print("\n" + "=" * 70)
        print(f"📊 BENCHMARK RESULTS - {r['variant']}")
        print("=" * 70)
        
        print(f"\n📦 Model Information:")
        print(f"   Variant: EfficientNet-{r['variant']}")
        print(f"   Weights: {r['model_path']}")
        
        print(f"\n📈 Processing Statistics:")
        print(f"   Frames Processed: {r['frames_processed']}/{r['num_frames']}")
        print(f"   Faces Detected: {r['faces_detected']} ({r['face_detection_rate']:.1%})")
        
        if 'avg_inference_ms' in r:
            print(f"\n⚡ Inference Performance:")
            print(f"   Average: {r['avg_inference_ms']:.2f} ms")
            print(f"   Std Dev: {r['std_inference_ms']:.2f} ms")
            print(f"   Min: {r['min_inference_ms']:.2f} ms")
            print(f"   Max: {r['max_inference_ms']:.2f} ms")
        
        if 'avg_fps' in r:
            print(f"\n🎥 FPS Performance:")
            print(f"   Average: {r['avg_fps']:.2f} FPS")
            print(f"   Std Dev: {r['std_fps']:.2f} FPS")
            print(f"   Min: {r['min_fps']:.2f} FPS")
            print(f"   Max: {r['max_fps']:.2f} FPS")
        
        if 'avg_face_detection_ms' in r:
            print(f"\n🔍 Component Breakdown:")
            print(f"   Face Detection:    {r['avg_face_detection_ms']:.2f} ms")
            if 'avg_facemesh_ms' in r:
                print(f"   FaceMesh + EAR/MAR: {r['avg_facemesh_ms']:.2f} ms")
            print(f"   Preprocessing:     {r['avg_preprocessing_ms']:.2f} ms")
            print(f"   Model Inference:   {r['avg_inference_ms']:.2f} ms")
            if 'avg_fusion_ms' in r:
                print(f"   Fusion+Stabilize:  {r['avg_fusion_ms']:.2f} ms")
            print(f"   Total Pipeline:    {r['avg_total_ms']:.2f} ms")
        
        print("\n" + "=" * 70)
    
    def save_results(self, output_dir="."):
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{self.results['variant']}_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"BENCHMARK RESULTS - EfficientNet-{self.results['variant']}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("MODEL INFORMATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Variant: EfficientNet-{self.results['variant']}\n")
            f.write(f"Weights: {self.results['model_path']}\n")
            f.write(f"Input Size: 224x224x3\n")
            f.write(f"Framework: TensorFlow/Keras\n\n")
            
            f.write("PROCESSING STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Frames Processed: {self.results['frames_processed']}/{self.results['num_frames']}\n")
            f.write(f"Faces Detected: {self.results['faces_detected']} ({self.results['face_detection_rate']:.1%})\n\n")
            
            if 'avg_inference_ms' in self.results:
                f.write("INFERENCE PERFORMANCE\n")
                f.write("-" * 70 + "\n")
                f.write(f"Average Inference Time: {self.results['avg_inference_ms']:.2f} ms\n")
                f.write(f"Standard Deviation: {self.results['std_inference_ms']:.2f} ms\n")
                f.write(f"Minimum: {self.results['min_inference_ms']:.2f} ms\n")
                f.write(f"Maximum: {self.results['max_inference_ms']:.2f} ms\n\n")
            
            if 'avg_fps' in self.results:
                f.write("FPS PERFORMANCE\n")
                f.write("-" * 70 + "\n")
                f.write(f"Average FPS: {self.results['avg_fps']:.2f}\n")
                f.write(f"Standard Deviation: {self.results['std_fps']:.2f}\n")
                f.write(f"Minimum FPS: {self.results['min_fps']:.2f}\n")
                f.write(f"Maximum FPS: {self.results['max_fps']:.2f}\n\n")
            
            if 'avg_face_detection_ms' in self.results:
                f.write("COMPONENT BREAKDOWN\n")
                f.write("-" * 70 + "\n")
                f.write(f"Face Detection:     {self.results['avg_face_detection_ms']:.2f} ms\n")
                if 'avg_facemesh_ms' in self.results:
                    f.write(f"FaceMesh + EAR/MAR: {self.results['avg_facemesh_ms']:.2f} ms\n")
                f.write(f"Preprocessing:      {self.results['avg_preprocessing_ms']:.2f} ms\n")
                f.write(f"Model Inference:    {self.results['avg_inference_ms']:.2f} ms\n")
                if 'avg_fusion_ms' in self.results:
                    f.write(f"Fusion+Stabilize:   {self.results['avg_fusion_ms']:.2f} ms\n")
                f.write(f"Total Pipeline:     {self.results['avg_total_ms']:.2f} ms\n\n")
            
            f.write("SYSTEM INFORMATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"TensorFlow Version: {tf.__version__}\n")
            f.write(f"OpenCV Version: {cv2.__version__}\n")
            f.write(f"NumPy Version: {np.__version__}\n")
            
        print(f"\n💾 Results saved to: {filepath}")
        return filepath


def main():
    """Run benchmarks for all available models"""
    models = []
    
    if os.path.exists("B0_16_batches.weights.keras"):
        models.append("B0_16_batches.weights.keras")
    
    if os.path.exists("B1_16_batches.weights.keras"):
        models.append("B1_16_batches.weights.keras")
    
    if not models:
        print("❌ No model weights found!")
        return
    
    print("🔬 Driver Drowsiness Detection - Benchmark Suite")
    print(f"   Found {len(models)} model(s) to benchmark\n")
    
    all_results = []
    
    for model_path in models:
        benchmark = BenchmarkRunner(model_path, num_frames=300)
        results = benchmark.run()
        
        if results:
            benchmark.print_results()
            benchmark.save_results()
            all_results.append(results)
        
        print("\n" + "=" * 70 + "\n")
        time.sleep(2)  # Brief pause between benchmarks
    
    # Comparison summary
    if len(all_results) > 1:
        print("\n📊 COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Model':<15} {'Avg Inference':<20} {'Avg FPS':<15} {'Detection Rate':<15}")
        print("-" * 70)
        for r in all_results:
            variant = r['variant']
            inf_ms = r.get('avg_inference_ms', 0)
            fps = r.get('avg_fps', 0)
            det_rate = r.get('face_detection_rate', 0)
            print(f"{variant:<15} {inf_ms:>8.2f} ms{'':<10} {fps:>6.2f}{'':<8} {det_rate:>6.1%}")
        print("=" * 70)


if __name__ == "__main__":
    main()
