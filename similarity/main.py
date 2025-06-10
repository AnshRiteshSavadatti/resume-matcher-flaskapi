from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Load the smallest performant model
bi_encoder = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2', device='cpu')

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body model
class ResumeRequest(BaseModel):
    resume: dict
    jobDescription: dict

# Resume preprocessing
def preprocess_resume(resume_json):
    parts = [
        resume_json.get("summary", ""),
        "Skills: " + ", ".join(resume_json.get("skills", []))
    ]
    for exp in resume_json.get("experience", []):
        description = exp.get("description", "")
        if isinstance(description, list):
            description = " ".join(description)
        parts.append(f"{exp.get('title', '')} at {exp.get('company', '')}: {description}")
    return " ".join(parts)

# Endpoint
@app.post("/compare")
async def compare_similarity(req: ResumeRequest):
    resume_text = preprocess_resume(req.resume)
    job_text = req.jobDescription.get("jobDescription", "")

    # Compute cosine similarity between sentence embeddings
    resume_embedding = bi_encoder.encode(resume_text, convert_to_tensor=True)
    job_embedding = bi_encoder.encode(job_text, convert_to_tensor=True)
    bi_score = 1 - cosine(resume_embedding.cpu(), job_embedding.cpu())

    # Boost and format score
    boosted_score = 10 + (100 - 10) * bi_score
    return {"similarityScore": float(round(boosted_score, 2))}
