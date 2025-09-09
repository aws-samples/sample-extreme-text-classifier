# Advanced Examples

This directory contains advanced examples that demonstrate the full capabilities of the multi-class text classifier, including PDF processing and reranking comparisons.

## Examples

### complete_example.py

A comprehensive example that demonstrates:

1. **Dataset Generation**: Creates a document types dataset with 30 classes
2. **Embedding Creation**: Generates and stores embeddings for fast similarity search
3. **PDF Content Extraction**: Extracts text and image descriptions from PDF files using Amazon Nova Lite
4. **Text Classification**: Classifies both regular text and PDF content
5. **Reranking Comparison**: Compares four different reranking approaches:
   - **No Reranking**: Pure similarity search baseline
   - **Amazon Rerank**: AWS native reranking service (fast, optimized)
   - **Cohere Rerank**: Third-party reranking via Bedrock (strong semantic understanding)
   - **Nova Lite Rerank**: Amazon's lightweight LLM with custom prompting (flexible)

#### Features Demonstrated

- **PDF Processing**: Extracts both text content and image descriptions from PDF files
- **Multi-modal Content**: Handles documents with both text and visual elements
- **Reranking Analysis**: Side-by-side comparison of different reranking methods
- **Performance Metrics**: Measures classification speed and accuracy
- **Error Handling**: Graceful fallback when reranking methods fail

#### Usage

```bash
python examples/advanced/complete_example.py
```

#### Requirements

- AWS credentials configured for Bedrock access
- All dependencies from requirements.txt
- PDF file in datasets folder (uses `datasets/dummy_invoice.pdf`)

#### Output

The example provides detailed output including:
- Dataset generation progress
- PDF extraction results (text length, images found)
- Classification results for multiple test cases
- **Performance statistics table** with end-to-end timing (semantic search + reranking) and throughput metrics for each method
- Reranking comparison table showing top 5 results from each method
- Agreement analysis between different reranking approaches
- Performance insights and recommendations

#### Reranking Methods Comparison

The example compares four approaches:

1. **No Reranking (Baseline)**
   - Uses pure cosine similarity from embeddings
   - Fast and consistent
   - Good baseline performance

2. **Amazon Rerank**
   - AWS native reranking service
   - Optimized for speed and efficiency
   - Good for production workloads

3. **Cohere Rerank**
   - Third-party service via AWS Bedrock
   - Strong semantic understanding
   - Excellent for complex text analysis

4. **Nova Lite Rerank**
   - Amazon's lightweight LLM
   - Flexible prompt-based approach
   - Good balance of cost and performance
   - Customizable reasoning process

#### Key Insights

- Different reranking methods may excel at different types of content
- Nova Lite offers the most flexibility through custom prompting
- Amazon Rerank provides the best speed/efficiency trade-off
- Cohere Rerank often provides the strongest semantic understanding
- Consider cost, latency, and accuracy trade-offs for your specific use case

#### Example Output

```
🚀 Multi-Class Text Classifier - Complete Example with PDF & Reranking
================================================================================

📋 Configuration:
   Domain: document types
   Number of classes: 30
   Dataset path: output/document_types_dataset.json
   Embeddings path: output/document_types_embeddings.pkl.gz
   PDF file: datasets/dummy_invoice.pdf

🔧 Step 1: Dataset Generation
----------------------------------------
✅ Dataset generated and saved with 30 classes

📄 Step 3: PDF Content Extraction
----------------------------------------
✅ PDF extraction completed in 2.45 seconds
📊 Extraction Results:
   Pages: 1
   Text length: 1247 characters
   Images found: 2

🔬 Step 5: Reranking Methods Comparison
--------------------------------------------------
⚡ Performance Statistics:
====================================================================================================
Method               Avg Time (s)    Classifications/sec  Status          Iterations  
----------------------------------------------------------------------------------------------------
No Reranking         0.125           8.00                 ✅ Success       3           
Amazon Rerank        0.972           1.03                 ✅ Success       3           
Cohere Rerank        1.359           0.74                 ✅ Success       3           
Nova Lite Rerank     2.281           0.44                 ✅ Success       3           

📈 Performance Insights:
   🚀 Fastest: No Reranking (0.125s)
   🐌 Slowest: Nova Lite Rerank (2.281s)
   ⚡ Speedup: 18.2x faster
   📊 Highest throughput: No Reranking (8.00 classifications/sec)
   📝 Note: Times include semantic search + reranking for end-to-end comparison

📊 Top 5 Results Comparison:
--------------------------------------------------------------------------------------------------------------------------
Rank No Reranking            Amazon Rerank            Cohere Rerank            Nova Lite Rerank        
--------------------------------------------------------------------------------------------------------------------------
1    Invoice (0.892)         Invoice (0.945)          Invoice (0.967)          Invoice (0.923)         
2    Bill (0.834)            Bill (0.876)             Receipt (0.845)          Receipt (0.887)         
3    Receipt (0.798)         Receipt (0.823)          Bill (0.812)             Bill (0.834)            
4    Financial Report (0.756) Financial Report (0.789) Financial Report (0.778) Financial Report (0.801)
5    Contract (0.723)        Contract (0.734)         Contract (0.745)         Contract (0.756)        

🎯 Top Result Analysis:
------------------------------------------------------------
   No Reranking:     Invoice ✅
   Amazon Rerank:    Invoice ✅
   Cohere Rerank:    Invoice ✅
   Nova Lite Rerank: Invoice ✅

📈 Agreement Analysis:
   🤝 All methods agree on the top result

💡 Performance Recommendations:
   🚀 For speed: No Reranking (0.125s total)
   📊 For throughput: No Reranking (8.00 classifications/sec)
   🎯 For accuracy: Consider Cohere Rerank for complex semantic understanding
   💰 For cost-effectiveness: No Reranking baseline or Nova Lite for custom logic
   📝 Note: Times include semantic search + reranking for end-to-end comparison
```

This example serves as a comprehensive demonstration of the classifier's capabilities and helps users understand the trade-offs between different reranking approaches.