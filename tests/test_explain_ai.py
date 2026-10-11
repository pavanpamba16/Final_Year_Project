import os
import sys
import unittest
from PIL import Image
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from explain_ai_engine import ExplainAIMultimodalOrchestrator, MalariaImageEngine, MalariaClinicalEngine

class TestExplainAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ExplainAIMultimodalOrchestrator()
        cls.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Test images
        cls.sample_parasitized = os.path.join(cls.base_dir, 'static', 'sample_smears', 'parasitized_sample_1.png')
        cls.sample_uninfected = os.path.join(cls.base_dir, 'static', 'sample_smears', 'uninfected_sample_1.png')
        
        # In case sample smears don't exist yet, find one from data/cell_images
        if not os.path.exists(cls.sample_parasitized):
            p_dir = os.path.join(cls.base_dir, 'data', 'cell_images', 'Parasitized')
            for f in os.listdir(p_dir):
                if f.endswith('.png'):
                    cls.sample_parasitized = os.path.join(p_dir, f)
                    break
        if not os.path.exists(cls.sample_uninfected):
            u_dir = os.path.join(cls.base_dir, 'data', 'cell_images', 'Uninfected')
            for f in os.listdir(u_dir):
                if f.endswith('.png'):
                    cls.sample_uninfected = os.path.join(u_dir, f)
                    break

    def test_01_image_only_mode_a(self):
        """Test 1: Image-only input (Mode A)"""
        print("\n--- TEST 1: IMAGE-ONLY INPUT (MODE A) ---")
        self.assertTrue(os.path.exists(self.sample_parasitized), "Parasitized test image must exist")
        res = self.orchestrator.evaluate_mode_a(self.sample_parasitized)
        
        self.assertEqual(res['mode'], 'Mode A (Image Only)')
        self.assertIn('image_result', res)
        img_res = res['image_result']
        self.assertNotIn('error', img_res)
        self.assertIn(img_res['predicted_class'], ['Parasitized', 'Uninfected'])
        self.assertGreaterEqual(img_res['confidence'], 50.0)
        self.assertIsNotNone(img_res['grad_cam_b64'])
        self.assertTrue(img_res['grad_cam_b64'].startswith('data:image/png;base64,'))
        print(f"Result: Class={img_res['predicted_class']}, Confidence={img_res['confidence']}%, Grad-CAM Generated=True")
        print("TEST 1 PASSED")

    def test_02_clinical_only_mode_b(self):
        """Test 2: Clinical-only input (Mode B)"""
        print("\n--- TEST 2: CLINICAL-ONLY INPUT (MODE B) ---")
        severe_patient = {
            'age': 35,
            'sex': 1,
            'fever': 1,
            'cold': 1,
            'rigor': 1,
            'fatigue': 1,
            'headace': 1,
            'bitter_tongue': 0,
            'vomitting': 1,
            'diarrhea': 0,
            'Convulsion': 1,
            'Anemia': 1,
            'jundice': 1,
            'cocacola_urine': 1,
            'hypoglycemia': 0,
            'prostraction': 1,
            'hyperpyrexia': 1
        }
        res = self.orchestrator.evaluate_mode_b(severe_patient)
        self.assertEqual(res['mode'], 'Mode B (Clinical Only)')
        clin_res = res['clinical_result']
        self.assertNotIn('error', clin_res)
        self.assertEqual(clin_res['target_col'], 'severe_maleria')
        self.assertTrue(clin_res['is_severe'])
        self.assertGreaterEqual(clin_res['probability'], 50.0)
        self.assertIsNotNone(clin_res['shap_figure_b64'])
        print(f"Result: Target={clin_res['target_col']}, Is_Severe={clin_res['is_severe']}, Prob={clin_res['probability']}%, SHAP Generated=True")
        print("TEST 2 PASSED")

    def test_03_dual_agreeing_mode_c(self):
        """Test 3: Both inputs with agreeing displayed labels (Mode C)"""
        print("\n--- TEST 3: DUAL INPUT WITH AGREEING LABELS (MODE C) ---")
        # Uninfected image + Mild/Negative clinical symptoms (Both negative)
        mild_patient = {
            'age': 28, 'sex': 0, 'fever': 0, 'cold': 0, 'rigor': 0, 'fatigue': 0,
            'headace': 0, 'bitter_tongue': 0, 'vomitting': 0, 'diarrhea': 0,
            'Convulsion': 0, 'Anemia': 0, 'jundice': 0, 'cocacola_urine': 0,
            'hypoglycemia': 0, 'prostraction': 0, 'hyperpyrexia': 0
        }
        res = self.orchestrator.evaluate_mode_c(self.sample_uninfected, mild_patient)
        self.assertFalse(res['has_conflict'], "Agreeing polarities should not trigger conflict warning")
        self.assertFalse(res['image_result']['is_parasitized'])
        self.assertFalse(res['clinical_result']['is_severe'])
        self.assertIsNone(res['conflict_title'])
        # Modality 2 only explainability in Mode C:
        self.assertEqual(res['shap_figure_b64'], res['clinical_result']['shap_figure_b64'])
        self.assertEqual(res['lime_figure_b64'], res['clinical_result']['lime_figure_b64'])
        self.assertIsNotNone(res['clinical_result']['shap_figure_b64'])
        self.assertIsNotNone(res['clinical_result']['lime_figure_b64'])
        # Overall result for concordant multimodal prediction:
        self.assertIsNotNone(res.get('overall_result'))
        self.assertFalse(res['overall_result']['is_severe'])
        self.assertEqual(res['overall_result']['heading'], 'Non-Severe Presentation (Negative)')
        print(f"Result: Image={res['image_result']['predicted_class']} ({res['image_result']['confidence']}%), Clinical Is_Severe={res['clinical_result']['is_severe']} ({res['clinical_result']['probability']}%), Has_Conflict={res['has_conflict']}")
        print("Modality 2 SHAP & LIME Verified for Mode C")
        print("TEST 3 PASSED")

    def test_04_dual_disagreeing_mode_c(self):
        """Test 4: Both inputs with disagreeing displayed labels (Mode C)"""
        print("\n--- TEST 4: DUAL INPUT WITH DISAGREEING LABELS (MODE C) ---")
        # Uninfected image (Negative) + Severe Clinical Symptoms (Positive)
        severe_patient = {
            'age': 40, 'sex': 1, 'fever': 1, 'cold': 1, 'rigor': 1, 'fatigue': 1,
            'headace': 1, 'bitter_tongue': 1, 'vomitting': 1, 'diarrhea': 1,
            'Convulsion': 1, 'Anemia': 1, 'jundice': 1, 'cocacola_urine': 1,
            'hypoglycemia': 1, 'prostraction': 1, 'hyperpyrexia': 1
        }
        res = self.orchestrator.evaluate_mode_c(self.sample_uninfected, severe_patient)
        self.assertTrue(res['has_conflict'], "Disagreeing polarities must trigger conflict")
        self.assertEqual(res['conflict_title'], "Conflicting Predictions — Further Assessment Recommended")
        self.assertEqual(res['conflict_message'], "The image and clinical predictions do not agree. Please verify the available information and seek appropriate medical assessment.")
        print(f"Result: Has_Conflict={res['has_conflict']}")
        print(f"Title: {res['conflict_title']}")
        print(f"Message: {res['conflict_message']}")
        print("TEST 4 PASSED")

    def test_05_missing_or_incomplete_inputs(self):
        """Test 5: Missing image or incomplete clinical inputs"""
        print("\n--- TEST 5: MISSING / INCOMPLETE INPUTS ---")
        # Mode B with invalid age
        invalid_clinical = {'age': -15, 'sex': 0}
        res_clin = self.orchestrator.evaluate_mode_b(invalid_clinical)
        self.assertIn('error', res_clin['clinical_result'])
        print(f"Clinical Validation Error Handled: {res_clin['clinical_result']['error']}")
        
        # Mode B with non-integer age
        invalid_clinical_2 = {'age': 'twenty-five', 'sex': 0}
        res_clin_2 = self.orchestrator.evaluate_mode_b(invalid_clinical_2)
        self.assertIn('error', res_clin_2['clinical_result'])
        print(f"Clinical Non-integer Handled: {res_clin_2['clinical_result']['error']}")
        
        # Mode B with missing age
        missing_age = {'sex': 0}
        res_missing_age = self.orchestrator.evaluate_mode_b(missing_age)
        self.assertIn('error', res_missing_age)
        self.assertIn('age', res_missing_age['error'].lower())
        print(f"Missing Age Handled: {res_missing_age['error']}")
        
        # Mode B with missing gender
        missing_sex = {'age': 25}
        res_missing_sex = self.orchestrator.evaluate_mode_b(missing_sex)
        self.assertIn('error', res_missing_sex)
        self.assertIn('gender', res_missing_sex['error'].lower())
        print(f"Missing Gender Handled: {res_missing_sex['error']}")
        print("TEST 5 PASSED")

    def test_06_invalid_or_unsupported_image(self):
        """Test 6: Invalid or unsupported image files"""
        print("\n--- TEST 6: INVALID / UNSUPPORTED IMAGE FILES ---")
        dummy_bad_file = os.path.join(self.base_dir, 'models', 'corrupt_test_file.xyz')
        with open(dummy_bad_file, 'w') as f:
            f.write("This is a plain text file, not a valid PNG or JPG image.")
            
        res = self.orchestrator.evaluate_mode_a(dummy_bad_file)
        if os.path.exists(dummy_bad_file):
            os.remove(dummy_bad_file)
            
        self.assertIn('error', res['image_result'])
        print(f"Invalid Image Error Handled Gracefully: {res['image_result']['error']}")
        print("TEST 6 PASSED")

    def test_07_model_persistence_and_restart(self):
        """Test 7: Model loading and preprocessing after application restart"""
        print("\n--- TEST 7: MODEL LOADING & PREPROCESSING AFTER APPLICATION RESTART ---")
        # Re-instantiate fresh engines simulating server reboot
        new_image_engine = MalariaImageEngine()
        new_clinical_engine = MalariaClinicalEngine()
        
        self.assertIsNotNone(new_image_engine.model, "MobileNetV2 model must reload successfully")
        self.assertIsNotNone(new_clinical_engine.model, "Clinical Voting Ensemble must reload successfully")
        self.assertEqual(len(new_clinical_engine.features), 17)
        self.assertIn('accuracy', new_image_engine.metrics)
        self.assertIn('roc_auc', new_clinical_engine.test_metrics)
        print(f"Image Model Reloaded: {new_image_engine.metrics}")
        print(f"Clinical Model Reloaded: ROC-AUC={new_clinical_engine.test_metrics.get('roc_auc')}")
        print("TEST 7 PASSED")

if __name__ == '__main__':
    unittest.main()
