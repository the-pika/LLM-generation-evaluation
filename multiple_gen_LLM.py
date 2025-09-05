import json
from transformers import AutoTokenizer, pipeline
import torch
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain

# Model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load pipeline ONCE
hf_pipeline = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda:0",
    # max_length=2000,
    do_sample=True,
    top_k=20,
    repetition_penalty=1.2,
    max_new_tokens=600,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

# LangChain wrapper with mutable generation config
model_kwargs = {"temperature": 0.5, "top_p": 0.9}  # Start temperature at 0.5
llm = HuggingFacePipeline(pipeline=hf_pipeline, model_kwargs=model_kwargs)

# Prompt template
template = """
You are an expert medical summarization assistant.

Summarize the following hospital discharge note in about 200 words, in clear and professional language suitable for both patients and clinicians. 
Be concise but include diagnosis, treatment, medications, and follow-up instructions.
Do not explain the summary. Do not include section headers or lists. Do not restate the task or say "I hope this helps".

{text}

SUMMARY:
"""
prompt = PromptTemplate(template=template, input_variables=["text"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Discharge note text
text = '''

Allergies: No Known Allergies / Adverse Drug Reactions

Chief Complaint: Epistaxis

Major Surgical or Invasive Procedure: None

History of Present Illness:
Mr. Abc came with a history of AAA s/p repair complicated by MI, hypertension, and hyperlipidemia who presents upon transfer from outside hospital with nasal fractures and epistaxis secondary to fall. While coughing, he tripped on the curb and suffered trauma to his face. He had no loss of consciousness. However, he had a persistent nosebleed and appeared to have some trauma to his face, thus was transferred for further care. There, a CT scan of the head, neck, and face were remarkable for a nasal bone and septal fracture. Given persistent epistaxis, bilateral Rhinorockets were placed. He had a small abrasion to the bridge of his nose which was not closed. Bleeding was well controlled. While in the OSH ED, he had an episode of nausea and coughed up some blood. At that time, he began to feel lightheaded and was noted to be hypotensive and bradycardic. Per report, he had a brief loss of consciousness, though quickly returned to his baseline. His family noted that his eyes rolled back into his head. The patient recalls the event and denies post-event confusion. He had no further episodes of syncope or hemodynamic changes.  Given the syncopal event and epistaxis, the patient was transferred for further care.

In the ED, initial vital signs 98.9 92 140/77 18 100%/RA. Labs were notable for WBC 11.3 (91%N), H/H 14.1/40.2, plt 147, BUN/Cr 36/1.5. HCTs were repeated which were stable. A urinalysis was negative. A CXR demonstrated a focal consolidation at the left lung base, possibly representing aspiration or developing pneumonia. The patient was given Tdap, amoxicillin-clavulanate for antibiotic prophylaxis, ondansetron, 500cc NS, and metoprolol tartrate 50mg. Clopidogrel was held.

IMAGING: PA and lateral views of the chest provided. The lungs are adequately aerated. There is a focal consolidation at the left lung base adjacent to the lateral hemidiaphragm. There is mild vascular engorgement. There is bilateral apical pleural thickening. The cardio mediastinal silhouette is remarkable for aortic arch calcifications. The heart is top normal in size.

ECHO - The left atrium is mildly dilated. Left ventricular wall thicknesses and cavity size are normal. There is mild regional left ventricular systolic dysfunction with focal apical hypokinesis. The remaining segments contract normally (LVEF = 55 %). No masses or thrombi are seen in the left ventricle. Right ventricular chamber size and free wall motion are normal. There are three aortic valve leaflets. There is mild aortic valve stenosis (valve area 1.7cm2). Mild (1+) aortic regurgitation is seen. The mitral valve leaflets are mildly thickened. Trivial mitral regurgitation is seen. The pulmonary artery systolic pressure could not be determined. There is a trivial/physiologic pericardial effusion.

IMPRESSION: Normal left ventricular cavity size with mild regional systolic dysfunction most c/w CAD (distal LAD distribution). Mild aortic valve stenosis. Mild aortic regurgitation.

Brief Hospital Course:
Mr. Abc came with history of AAA s/p repair complicated by MI, hypertension, and hyperlipidemia who presented with nasal fractures and epistaxis after mechanical fall with hospital course complicated by NSTEMI. Patient presenting after mechanical fall with Rhinorockets placed at outside hospital for ongoing epistaxis. CT scan from that hospital demonstrated nasal bone and septal fractures. The Rhinorockets were maintained while inpatient and discontinued prior to discharge. He was encouraged to use oxymetolazone nasal spray and hold pressure should bleeding reoccur.

Patient found to have mild elevation of troponin in the ED. This was trended and eventually rose to 1.5, though MB component downtrended during course of admission. The patient was without chest pain or other cardiac symptoms. Cardiology was consulted who thought that this was most likely secondary to demand ischemia (type II MI) secondary to his fall. An echocardiogram demonstrated aortic stenosis and likely distal LAD disease based on wall motion abnormalities. The patient's metoprolol was uptitrated, his pravastatin was converted to atorvastatin, his clopidogrel was maintained, and he was started on aspirin.

Patient reported to be mildly hypoxic in the ED, though he maintained normal oxygen saturations on room air. He denied shortness of breath or cough, fevers, or other infectious symptoms and had no leukocytosis. A CXR revealed consolidation in left lung, thought to be possibly related to aspirated blood. -monitor O2 saturation, temperature, trend WBC. He was convered with antibiotics while inpatient as he required prophylaxis for the Rhinorockets, but this was discontinued upon discharge.

Acute kidney injury: Patient presented with creatinine of 1.5 with last creatinine at PCP 1.8. Patient was unaware of a history of kidney disease. The patient was discharged with a stable creatinine.

Peripheral vascular disease: Patient had a history of AAA repair in ___ without history of MI per PCP. Patient denied history of CABG or cardiac/peripheral stents. A cardiac regimen was continued, as above.

Discharge Disposition: Home With Service

Discharge Diagnosis: Nasal fracture, Epistaxis, NSTEMI

Discharge Condition:
Mental Status: Clear and coherent.
Level of Consciousness: Alert and interactive.
Activity Status: Ambulatory - Independent.

Discharge Instructions:
Mr. Abc, you were admitted after you fell and broke your nose. You had nose bleeds that were difficult to control, thus plugs were placed in your nose to stop the bleeding. During your hospital course, you were found to have high troponins, a blood test for the heart. A ultrasound of your heart was performed. You should follow-up with your PCP to discuss stress test.

'''

def generate_and_save_summaries(text, n=10, txt_file="diverse_summaries.txt", json_file="diverse_summaries.json"):
    summaries = []
    json_data = []

    for i in range(n):
        temperature = 0.5 + 0.05 * i  # 🔁 start from 0.5
        top_p = 0.9 if i % 2 == 0 else 0.8

        # Update generation settings
        llm.model_kwargs["temperature"] = temperature
        llm.model_kwargs["top_p"] = top_p

        summary = llm_chain.run(text).strip()
        summaries.append((i + 1, temperature, top_p, summary))

        # Save summary with metadata for JSON
        json_data.append({
            "summary_id": i + 1,
            "temperature": round(temperature, 2),
            "top_p": round(top_p, 2),
            "summary_text": summary,
            "clarity_score": None,      # to be filled during evaluation
            "accuracy_score": None,
            "completeness_score": None,
            "coherence_score": None
        })

    # Save summaries to .txt file (without prompt)
    with open(txt_file, "w", encoding="utf-8") as f:
        for idx, temp, p, s in summaries:
            f.write(f"Summary {idx} (temperature={temp:.2f}, top_p={p}):\n{s}\n{'-'*80}\n\n")
    print(f"Saved summaries to {txt_file}")

    # Save detailed metadata to .json
    with open(json_file, "w", encoding="utf-8") as jf:
        json.dump(json_data, jf, indent=2, ensure_ascii=False)
    print(f"Saved structured metadata to {json_file}")

# Run it
generate_and_save_summaries(text)
