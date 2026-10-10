import os
import torch
from PIL import Image
from torchvision import transforms
try:
    from train_vision import MalariaVisionCNN
except ImportError:
    from src.train_vision import MalariaVisionCNN

class MalariaMultiModalEngine:
    def __init__(self, vision_model_path=None):
        if vision_model_path is None:
            vision_model_path = r"C:\Users\pamba\OneDrive\Desktop\Final-Project\models\vision\malaria_vision_cnn.pt"
            
        self.device = torch.device('cpu')
        self.model = None
        self.metrics = None
        self.img_size = 64
        
        if os.path.exists(vision_model_path):
            try:
                pkg = torch.load(vision_model_path, map_location=self.device)
                self.model = MalariaVisionCNN(num_classes=2)
                self.model.load_state_dict(pkg['state_dict'])
                self.model.eval()
                self.metrics = pkg.get('metrics', {})
                self.img_size = pkg.get('img_size', 64)
                print(f"[Multi-Modal] Loaded Vision CNN Model from {vision_model_path}")
            except Exception as e:
                print(f"[Multi-Modal] Error loading Vision CNN: {e}")
                
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def analyze_blood_smear(self, image_input):
        """
        Runs microscopic blood smear image inference.
        image_input can be a file path, file object, or PIL Image.
        """
        if self.model is None:
            return None
            
        try:
            if isinstance(image_input, str):
                pil_img = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, Image.Image):
                pil_img = image_input.convert('RGB')
            else:
                pil_img = Image.open(image_input).convert('RGB')
                
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                
            parasite_prob = float(probs[1].item())
            is_parasitized = parasite_prob >= 0.5
            confidence = round((parasite_prob if is_parasitized else (1.0 - parasite_prob)) * 100, 1)
            
            return {
                'parasite_detected': is_parasitized,
                'status': 'Parasitized (Infected Erythrocyte)' if is_parasitized else 'Uninfected (Healthy Erythrocyte)',
                'badge': 'Positive (Parasitized)' if is_parasitized else 'Negative (Uninfected)',
                'probability': round(parasite_prob * 100, 1),
                'confidence': confidence,
                'vision_metrics': self.metrics
            }
        except Exception as e:
            print(f"[Multi-Modal] Error analyzing blood smear image: {e}")
            return None

    def fuse_predictions(self, clinical_prob, vision_result, active_severe_count, active_mild_count):
        """
        Mathematically fuses microscopic vision diagnosis with clinical symptom severity.
        """
        if vision_result is None:
            # Standalone clinical triage
            return {
                'fused_probability': clinical_prob,
                'modality_used': 'Clinical Symptoms Only',
                'vision_result': None,
                'clinical_probability': clinical_prob
            }
            
        p_vision = vision_result['probability'] / 100.0
        parasite_detected = vision_result['parasite_detected']
        
        if parasite_detected:
            # Laboratory smear confirmed positive for Plasmodium
            if active_severe_count >= 1:
                # Confirmed infection + WHO cardinal severe complications = High Risk Severe Malaria
                fused_prob = max(0.78, min(0.985, clinical_prob * 0.65 + p_vision * 0.30 + 0.05))
                diagnostic_synthesis = "Confirmed Severe Malaria (Microscopy Positive + WHO Severe Complications Present)"
                triage_path = "EMERGENCY: Immediate Parenteral Artesunate (IV/IM) & Inpatient ICU Monitoring"
            elif active_mild_count > 0:
                # Confirmed infection + mild symptoms = Uncomplicated Malaria
                fused_prob = min(0.38, max(0.25, clinical_prob * 0.70 + p_vision * 0.15))
                diagnostic_synthesis = "Confirmed Uncomplicated Malaria (Microscopy Positive, No Organ Complications)"
                triage_path = "OUTPATIENT: Oral Artemisinin-based Combination Therapy (ACT, e.g. Coartem)"
            else:
                # Confirmed infection + asymptomatic
                fused_prob = 0.22
                diagnostic_synthesis = "Sub-clinical / Early Parasitemia Detected via Microscopy"
                triage_path = "OUTPATIENT: Antimalarial treatment to prevent clinical progression"
        else:
            # Laboratory smear negative for Plasmodium
            if active_severe_count >= 1:
                # Smear negative but severe symptoms present -> Non-malarial acute emergency!
                fused_prob = 0.25
                diagnostic_synthesis = "Severe Acute Non-Malarial Syndrome (Microscopy Negative for Plasmodium)"
                triage_path = "URGENT DIFFERENTIAL WORKUP: Investigate Bacterial Sepsis, Meningitis, or Viral Encephalitis"
            elif active_mild_count > 0:
                # Smear negative with mild fever/headache -> Non-malarial febrile illness
                fused_prob = 0.10
                diagnostic_synthesis = "Non-Malarial Febrile Illness (Microscopy Negative for Plasmodium)"
                triage_path = "OUTPATIENT: Evaluate for Viral Syndrome, Typhoid, or Upper Respiratory Infection"
            else:
                # Smear negative and asymptomatic
                fused_prob = 0.05
                diagnostic_synthesis = "Healthy Baseline (Microscopy Negative, No Manifestations)"
                triage_path = "ROUTINE: Normal healthy profile"
                
        return {
            'fused_probability': fused_prob,
            'modality_used': 'Multi-Modal Fusion (Microscopic Smear Vision + Clinical Manifestations)',
            'vision_result': vision_result,
            'clinical_probability': clinical_prob,
            'diagnostic_synthesis': diagnostic_synthesis,
            'triage_path': triage_path
        }
