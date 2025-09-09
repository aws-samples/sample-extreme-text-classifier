"""
Document classification with reranking using Amazon Rerank.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from multi_class_text_classifier import TextClassifier
from multi_class_text_classifier.models.data_models import RerankingConfig

print("Document Classification with Reranking")
print("=" * 50)

# Configure reranking with Amazon Rerank (fastest and most cost-effective)
reranking_config = RerankingConfig(
    model_type="amazon_rerank",
    model_id="arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0",
    top_k_candidates=5,
    aws_region="us-west-2"
)

# Alternative reranking configurations (uncomment to try):

# # Cohere Rerank
# reranking_config = RerankingConfig(
#     model_type="cohere_rerank",
#     model_id="cohere.rerank-v3-5:0",
#     top_k_candidates=5,
#     aws_region="us-west-2",
#     model_parameters={
#         "top_n": 5,
#         "return_documents": True
#     }
# )

# # Nova Lite LLM
# reranking_config = RerankingConfig(
#     model_type="llm",
#     model_id="us.amazon.nova-lite-v1:0",
#     top_k_candidates=5,
#     aws_region="us-west-2",
#     model_parameters={
#         "temperature": 0.1,
#         "max_tokens": 1000
#     }
# )

# Initialize classifier with reranking - only need embeddings path since dataset was already processed in previous example
classifier = TextClassifier(
    embeddings_path="datasets/document_classification_embeddings.pkl.gz",
    reranking_config=reranking_config
)

# Test text that might benefit from reranking
text = "This quarterly financial report shows revenue growth of 15% compared to last year, with detailed analysis of market trends and performance metrics."

print(f"Text: {text}")
print()

result = classifier.predict(text)

print(f"Reranked: {result.reranked}")  # True if reranking was applied
print(f"Predicted Class: {result.predicted_class.name}")
print(f"Confidence: {result.effective_score:.4f}")
print(f"Description: {result.predicted_class.description}")

# Show rerank scores if available
print("\nTop alternatives with similarity and rerank scores:")
for alt in result.alternatives[1:3]:
    rerank_info = f", rerank={alt.rerank_score:.3f}" if alt.rerank_score is not None else ""
    print(f"  {alt.class_definition.name}: similarity={alt.similarity_score:.3f}{rerank_info}")
