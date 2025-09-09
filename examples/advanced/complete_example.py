#!/usr/bin/env python3
"""
Complete Example: Multi-Class Text Classifier with PDF Processing and Reranking Comparison

This example demonstrates the full workflow of the multi-class text classifier:
1. Dataset generation for a specific domain
2. Embedding creation and storage
3. Text classification with detailed results
4. PDF content extraction and classification
5. Comparison between different reranking methods (No rerank, Amazon, Cohere, Nova Lite)

Usage:
    python examples/advanced/complete_example.py

Requirements:
    - Python 3.8+
    - All dependencies from requirements.txt
    - AWS credentials configured for Bedrock access
"""

import sys
import time
import os
from pathlib import Path

# Add the parent directory to the path so we can import the classifier
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from multi_class_text_classifier import (
    DatasetGenerator, 
    TextClassifier,
    PDFExtractor,
    PDFExtractorConfig,
    create_pdf_extractor
)
from multi_class_text_classifier.models.data_models import RerankingConfig
from multi_class_text_classifier.llm_reranker import LLMReranker
from multi_class_text_classifier.dataset_loader import ClassesDataset
from multi_class_text_classifier.similarity_search import SimilaritySearch
from multi_class_text_classifier.exceptions import ClassifierError


def main():
    """
    Complete example demonstrating dataset generation, embedding creation, PDF processing, and reranking comparison.
    """
    print("🚀 Multi-Class Text Classifier - Complete Example with PDF & Reranking")
    print("=" * 80)
    
    # Configuration
    domain = "document types"
    num_classes = 30
    dataset_path = "output/document_types_dataset.json"
    embeddings_path = "output/document_types_embeddings.pkl.gz"
    pdf_path = "datasets/dummy_invoice.pdf"
    
    print(f"📋 Configuration:")
    print(f"   Domain: {domain}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Dataset path: {dataset_path}")
    print(f"   Embeddings path: {embeddings_path}")
    print(f"   PDF file: {pdf_path}")
    print()
    
    # Step 1: Generate Dataset
    print("🔧 Step 1: Dataset Generation")
    print("-" * 40)
    
    try:
        # Check if dataset already exists
        if Path(dataset_path).exists():
            print(f"✅ Dataset already exists at {dataset_path}")
        else:
            print("🔄 Generating new dataset...")
            generator = DatasetGenerator()
            dataset = generator.generate_dummy_dataset(domain=domain, num_classes=num_classes)
            generator.save_dataset(dataset, dataset_path)
            print(f"✅ Dataset generated and saved with {len(dataset['classes'])} classes")
            
            # Show sample classes
            print("\n📝 Sample classes generated:")
            for i, class_def in enumerate(dataset['classes'][:3]):
                print(f"   {i+1}. {class_def['name']}")
                print(f"      {class_def['description'][:70]}...")
        
    except Exception as e:
        print(f"❌ Dataset generation failed: {e}")
        return 1
    
    print()
    
    # Step 2: Initialize Classifier and Generate Embeddings
    print("🧠 Step 2: Classifier Initialization & Embedding Generation")
    print("-" * 60)
    
    try:
        print("🔄 Initializing text classifier...")
        start_time = time.time()
        
        classifier = TextClassifier(
            dataset_path=dataset_path,
            embeddings_path=embeddings_path
        )
        
        # This will automatically generate embeddings if they don't exist
        embedding_info = classifier.get_embedding_info()
        
        init_time = time.time() - start_time
        print(f"✅ Classifier initialized in {init_time:.2f} seconds")
        
        print(f"\n📊 Embedding Information:")
        print(f"   Total classes: {embedding_info['total_classes']}")
        print(f"   Embedding dimension: {embedding_info['embedding_dimension']}")
        print(f"   Classes with embeddings: {embedding_info['classes_with_embeddings']}")
        print(f"   Dataset path: {embedding_info['dataset_path']}")
        print(f"   Embeddings path: {embedding_info['embeddings_path']}")
        
    except ClassifierError as e:
        print(f"❌ Classifier initialization failed: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error during initialization: {e}")
        return 1
    
    print()
    
    # Step 3: PDF Content Extraction
    print("📄 Step 3: PDF Content Extraction")
    print("-" * 40)
    
    pdf_content = None
    try:
        if not Path(pdf_path).exists():
            print(f"❌ PDF file not found: {pdf_path}")
            print("💡 Skipping PDF processing")
        else:
            print(f"🔄 Extracting content from PDF: {pdf_path}")
            
            # Create PDF extractor with Nova Lite
            extractor = create_pdf_extractor(
                model_id="us.amazon.nova-lite-v1:0",
                extract_text=True,
                extract_images=True
            )
            
            start_time = time.time()
            content = extractor.extract_from_file(pdf_path)
            extraction_time = time.time() - start_time
            
            print(f"✅ PDF extraction completed in {extraction_time:.2f} seconds")
            print(f"📊 Extraction Results:")
            print(f"   Pages: {content.page_count}")
            print(f"   Text length: {len(content.text_content)} characters")
            print(f"   Images found: {len(content.images)}")
            
            # Show text preview
            if content.text_content:
                print(f"\n📝 Text Preview:")
                preview = content.text_content[:300] + "..." if len(content.text_content) > 300 else content.text_content
                print(f"   {preview}")
            
            # Show image descriptions
            if content.images:
                print(f"\n🖼️  Image Descriptions:")
                for img in content.images[:2]:  # Show first 2 images
                    print(f"   Page {img.page_number}, Image {img.image_index}: {img.description[:100]}...")
                    if img.ocr_text and img.ocr_text != "NA":
                        print(f"   OCR: {img.ocr_text[:100]}...")
            
            pdf_content = content.full_content
            print(f"\n✅ PDF content ready for classification ({len(pdf_content)} characters)")
            
    except Exception as e:
        print(f"❌ PDF extraction failed: {e}")
        print("💡 Continuing with text-only examples")
    
    print()
    
    # Step 4: Text Classification Examples
    print("🎯 Step 4: Text Classification Examples")
    print("-" * 45)
    
    # Test cases with different complexity levels
    test_cases = [
        {
            "text": "Invoice for professional services rendered in Q3 2024",
            "description": "Invoice document"
        },
        {
            "text": "Employment contract for software engineer position",
            "description": "Contract document"
        },
        {
            "text": "Monthly financial report showing revenue and expenses",
            "description": "Financial report"
        },
        {
            "text": "User manual for operating the new equipment",
            "description": "Technical documentation"
        },
        {
            "text": "Marketing brochure for new product launch",
            "description": "Marketing material"
        }
    ]
    
    # Add PDF content as a test case if available
    if pdf_content:
        test_cases.insert(0, {
            "text": pdf_content,
            "description": "Extracted PDF content"
        })
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {test_case['description']}")
        if test_case['description'] == "Extracted PDF content":
            print(f"📝 Input: PDF content ({len(test_case['text'])} characters)")
        else:
            print(f"📝 Input: \"{test_case['text']}\"")
        print("-" * 50)
        
        try:
            start_time = time.time()
            result = classifier.predict(
                text=test_case['text'],
                top_k=3
            )
            classification_time = time.time() - start_time
            
            print(f"🏆 Predicted Class: {result.predicted_class.name}")
            print(f"🎯 Confidence: {result.effective_score:.4f} ({result.get_confidence_level()})")
            print(f"⏱️  Classification Time: {classification_time:.3f} seconds")
            
            print(f"\n📋 Top {len(result.alternatives)} Alternatives:")
            for j, alt in enumerate(result.alternatives, 1):
                print(f"   {j}. {alt.class_definition.name}")
                print(f"      Effective score: {alt.effective_score:.4f}")
                print(f"      Description: {alt.class_definition.description[:60]}...")
        
        except Exception as e:
            print(f"❌ Classification failed: {e}")
        except Exception as e:
            print(f"❌ Unexpected error during classification: {e}")
    
    print()
    
    # Step 5: Reranking Methods Comparison
    print("� Stepp 5: Reranking Methods Comparison")
    print("-" * 50)
    
    if pdf_content:
        try:
            comparison_result = compare_reranking_methods(
                text=pdf_content,
                dataset_path=dataset_path,
                embeddings_path=embeddings_path
            )
            
            if comparison_result == 0:
                print("✅ Reranking comparison completed successfully")
            else:
                print("⚠️  Reranking comparison completed with some issues")
                
        except Exception as e:
            print(f"❌ Reranking comparison failed: {e}")
    else:
        print("⚠️  Skipping reranking comparison (no PDF content available)")
    
    print()
    
    # Step 6: Performance Analysis
    print("📈 Step 6: Performance Analysis")
    print("-" * 35)
    
    try:
        # Measure classification performance with multiple texts
        performance_texts = [
            "Invoice for consulting services",
            "Employment agreement document",
            "Financial quarterly report",
            "Technical user manual",
            "Marketing presentation slides"
        ]
        
        print(f"🔄 Running performance test with {len(performance_texts)} texts...")
        
        total_time = 0
        successful_classifications = 0
        
        for text in performance_texts:
            try:
                start_time = time.time()
                result = classifier.predict(text)
                classification_time = time.time() - start_time
                total_time += classification_time
                successful_classifications += 1
            except Exception:
                pass  # Skip failed classifications for performance measurement
        
        if successful_classifications > 0:
            avg_time = total_time / successful_classifications
            print(f"✅ Performance Results:")
            print(f"   Successful classifications: {successful_classifications}/{len(performance_texts)}")
            print(f"   Average classification time: {avg_time:.3f} seconds")
            print(f"   Total time: {total_time:.3f} seconds")
            print(f"   Classifications per second: {successful_classifications/total_time:.2f}")
        else:
            print("❌ No successful classifications for performance measurement")
    
    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")
    
    print()
    
    # Step 7: Summary
    print("🎉 Step 7: Summary")
    print("-" * 20)
    
    print("✅ Complete example finished successfully!")
    print(f"📊 Classifier loaded with {embedding_info['total_classes']} classes")
    print(f"📁 Dataset saved at: {dataset_path}")
    print(f"🧠 Embeddings saved at: {embeddings_path}")
    if pdf_content:
        print(f"📄 PDF processed: {pdf_path}")
    print("\n💡 Next steps:")
    print("   - Try classifying your own text with classifier.predict(your_text)")
    print("   - Experiment with different domains by generating new datasets")
    print("   - Test different reranking methods for improved accuracy")
    print("   - Process your own PDF documents")
    print("   - Check the getting_started examples for more features")
    
    return 0


def compare_reranking_methods(text: str, dataset_path: str, embeddings_path: str) -> int:
    """
    Compare different reranking methods side by side.
    
    Args:
        text: Text to classify and compare reranking methods on
        dataset_path: Path to the dataset file
        embeddings_path: Path to the embeddings file
        
    Returns:
        0 if successful, 1 if failed
    """
    print("🔬 Reranking Methods Comparison")
    print("=" * 60)
    
    try:
        # Load dataset with embeddings
        print("🔄 Loading dataset for reranking comparison...")
        dataset = ClassesDataset()
        classes = dataset.load_dataset_with_embeddings(embeddings_path)
        print(f"✅ Dataset loaded with {len(classes)} classes")
        
        # Initialize similarity search
        similarity_search = SimilaritySearch(classes)
        
        # Generate embedding for test text
        test_embedding = dataset.generate_text_embedding(text)
        
        # Get similarity search results (baseline)
        similarity_candidates = similarity_search.find_similar_classes(test_embedding, top_k=8)
        
        print(f"\n📝 Test Text: PDF content ({len(text)} characters)")
        print("=" * 80)
        
        # Configure reranking methods
        reranking_configs = {
            "Amazon Rerank": RerankingConfig(
                model_type="amazon_rerank",
                model_id="arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0",
                top_k_candidates=5,
                aws_region="us-west-2",
                fallback_on_error=True
            ),
            "Cohere Rerank": RerankingConfig(
                model_type="cohere_rerank",
                model_id="cohere.rerank-v3-5:0",
                top_k_candidates=5,
                aws_region="us-west-2",
                fallback_on_error=True
            ),
            "Nova Lite Rerank": RerankingConfig(
                model_type="llm",
                model_id="us.amazon.nova-lite-v1:0",
                top_k_candidates=5,
                aws_region="us-west-2",
                model_parameters={
                    "temperature": 0.1,
                    "max_tokens": 1000,
                    "top_p": 0.9
                },
                fallback_on_error=True
            )
        }
        
        # Apply each reranking method with performance measurement
        reranked_results = {}
        success_status = {}
        performance_stats = {}
        
        # Test multiple times for better statistics
        num_iterations = 3
        print(f"\n⏱️  Running {num_iterations} iterations for performance measurement...")
        print("📊 Note: Timing includes both semantic search + reranking for fair comparison")
        
        # Baseline: No reranking (similarity search only)
        print(f"🔄 Testing No Reranking (baseline)...")
        no_rerank_times = []
        for i in range(num_iterations):
            start_time = time.time()
            # Generate embedding and perform similarity search (the complete process)
            test_embedding = dataset.generate_text_embedding(text)
            baseline_candidates = similarity_search.find_similar_classes(test_embedding, top_k=8)
            baseline_result = baseline_candidates[:5]  # Top 5 like reranking
            end_time = time.time()
            no_rerank_times.append(end_time - start_time)
        
        avg_no_rerank_time = sum(no_rerank_times) / len(no_rerank_times)
        performance_stats["No Reranking"] = {
            "avg_time": avg_no_rerank_time,
            "classifications_per_second": 1.0 / avg_no_rerank_time if avg_no_rerank_time > 0 else 0,
            "success": True,
            "iterations": num_iterations
        }
        reranked_results["No Reranking"] = similarity_candidates
        success_status["No Reranking"] = True
        print(f"✅ No Reranking completed - Avg time: {avg_no_rerank_time:.3f}s")
        
        # Test each reranking method
        for method_name, config in reranking_configs.items():
            print(f"🔄 Testing {method_name}...")
            method_times = []
            method_success = True
            final_result = similarity_candidates
            
            try:
                reranker = LLMReranker(config)
                
                # Run multiple iterations for timing (including semantic search)
                for i in range(num_iterations):
                    start_time = time.time()
                    
                    # Include the semantic search time (same as baseline)
                    test_embedding = dataset.generate_text_embedding(text)
                    fresh_candidates = similarity_search.find_similar_classes(test_embedding, top_k=8)
                    
                    # Apply reranking
                    reranked = reranker.rerank_candidates(text, fresh_candidates)
                    
                    end_time = time.time()
                    method_times.append(end_time - start_time)
                    if i == 0:  # Use first result for display
                        final_result = reranked
                
                avg_time = sum(method_times) / len(method_times)
                performance_stats[method_name] = {
                    "avg_time": avg_time,
                    "classifications_per_second": 1.0 / avg_time if avg_time > 0 else 0,
                    "success": True,
                    "iterations": num_iterations
                }
                print(f"✅ {method_name} completed - Avg time: {avg_time:.3f}s")
                
            except Exception as e:
                print(f"❌ {method_name} failed: {e}")
                method_success = False
                performance_stats[method_name] = {
                    "avg_time": 0,
                    "classifications_per_second": 0,
                    "success": False,
                    "iterations": 0,
                    "error": str(e)
                }
            
            reranked_results[method_name] = final_result
            success_status[method_name] = method_success
        
        # Performance Statistics Table
        print(f"\n⚡ Performance Statistics:")
        print("=" * 100)
        print(f"{'Method':<20} {'Avg Time (s)':<15} {'Classifications/sec':<20} {'Status':<15} {'Iterations':<12}")
        print("-" * 100)
        
        for method_name in ["No Reranking", "Amazon Rerank", "Cohere Rerank", "Nova Lite Rerank"]:
            stats = performance_stats.get(method_name, {})
            avg_time = stats.get("avg_time", 0)
            cps = stats.get("classifications_per_second", 0)
            success = stats.get("success", False)
            iterations = stats.get("iterations", 0)
            
            status = "✅ Success" if success else "❌ Failed"
            time_str = f"{avg_time:.3f}" if avg_time > 0 else "N/A"
            cps_str = f"{cps:.2f}" if cps > 0 else "N/A"
            
            print(f"{method_name:<20} {time_str:<15} {cps_str:<20} {status:<15} {iterations:<12}")
        
        # Performance insights
        print(f"\n📈 Performance Insights:")
        successful_methods = [name for name, stats in performance_stats.items() if stats.get("success", False)]
        if len(successful_methods) > 1:
            # Find fastest and slowest
            times = [(name, stats["avg_time"]) for name, stats in performance_stats.items() if stats.get("success", False)]
            times.sort(key=lambda x: x[1])
            fastest = times[0]
            slowest = times[-1]
            
            speedup = slowest[1] / fastest[1] if fastest[1] > 0 else 1
            print(f"   🚀 Fastest: {fastest[0]} ({fastest[1]:.3f}s)")
            print(f"   🐌 Slowest: {slowest[0]} ({slowest[1]:.3f}s)")
            print(f"   ⚡ Speedup: {speedup:.1f}x faster")
            
            # Throughput comparison
            throughputs = [(name, stats["classifications_per_second"]) for name, stats in performance_stats.items() if stats.get("success", False)]
            throughputs.sort(key=lambda x: x[1], reverse=True)
            print(f"   📊 Highest throughput: {throughputs[0][0]} ({throughputs[0][1]:.2f} classifications/sec)")
        
        # Display comparison table
        print(f"\n📊 Top 5 Results Comparison:")
        print("-" * 130)
        header = f"{'Rank':<4} {'No Reranking':<25} {'Amazon Rerank':<25} {'Cohere Rerank':<25} {'Nova Lite Rerank':<25}"
        print(header)
        print("-" * 130)
        
        for rank in range(1, 6):
            row_parts = [f"{rank:<4}"]
            
            # Each method including no reranking
            for method_name in ["No Reranking", "Amazon Rerank", "Cohere Rerank", "Nova Lite Rerank"]:
                if rank <= len(reranked_results[method_name]):
                    candidate = reranked_results[method_name][rank-1]
                    name = candidate.class_definition.name[:18]
                    if method_name == "No Reranking":
                        score = f"({candidate.similarity_score:.3f})"
                    else:
                        score = f"({candidate.rerank_score:.3f})" if candidate.rerank_score is not None else f"({candidate.similarity_score:.3f})"
                    display = f"{name} {score}"
                else:
                    display = "N/A"
                row_parts.append(f"{display:<25}")
            
            print("".join(row_parts))
        
        # Top result analysis
        print(f"\n🎯 Top Result Analysis:")
        print("-" * 60)
        
        no_rerank_top = reranked_results["No Reranking"][0].class_definition.name
        amazon_top = reranked_results["Amazon Rerank"][0].class_definition.name
        cohere_top = reranked_results["Cohere Rerank"][0].class_definition.name
        nova_lite_top = reranked_results["Nova Lite Rerank"][0].class_definition.name
        
        print(f"   No Reranking:     {no_rerank_top} {'✅' if success_status['No Reranking'] else '❌'}")
        print(f"   Amazon Rerank:    {amazon_top} {'✅' if success_status['Amazon Rerank'] else '❌'}")
        print(f"   Cohere Rerank:    {cohere_top} {'✅' if success_status['Cohere Rerank'] else '❌'}")
        print(f"   Nova Lite Rerank: {nova_lite_top} {'✅' if success_status['Nova Lite Rerank'] else '❌'}")
        
        # Agreement analysis
        all_same = no_rerank_top == amazon_top == cohere_top == nova_lite_top
        rerankers_agree = amazon_top == cohere_top == nova_lite_top
        
        print(f"\n📈 Agreement Analysis:")
        if all_same:
            print("   🤝 All methods agree on the top result")
        elif rerankers_agree:
            print("   🤝 All reranking methods agree (differ from no reranking)")
        else:
            agreements = []
            if no_rerank_top == amazon_top:
                agreements.append("No Rerank & Amazon")
            if no_rerank_top == cohere_top:
                agreements.append("No Rerank & Cohere")
            if no_rerank_top == nova_lite_top:
                agreements.append("No Rerank & Nova Lite")
            if amazon_top == cohere_top:
                agreements.append("Amazon & Cohere")
            if amazon_top == nova_lite_top:
                agreements.append("Amazon & Nova Lite")
            if cohere_top == nova_lite_top:
                agreements.append("Cohere & Nova Lite")
            
            if agreements:
                print(f"   🤝 Partial agreement: {', '.join(agreements)}")
            else:
                print("   🔄 All methods disagree - each has a different top result")
        
        # Method summary with performance context
        print(f"\n🔍 Reranking Methods Summary:")
        print("=" * 80)
        print("📊 No Reranking (Baseline):")
        print("   • Pure cosine similarity from embeddings")
        print("   • Minimal computational overhead")
        if "No Reranking" in performance_stats:
            stats = performance_stats["No Reranking"]
            print(f"   • Performance: {stats['avg_time']:.3f}s avg, {stats['classifications_per_second']:.2f} classifications/sec")
        
        print("\n📊 Amazon Rerank:")
        print("   • AWS native reranking service")
        if "Amazon Rerank" in performance_stats:
            stats = performance_stats["Amazon Rerank"]
            if stats['success']:
                print(f"   • Performance: {stats['avg_time']:.3f}s avg, {stats['classifications_per_second']:.2f} classifications/sec")
            else:
                print(f"   • Performance: Failed - {stats.get('error', 'Unknown error')}")
        
        print("\n📊 Cohere Rerank:")
        print("   • Third-party service via Bedrock")
        if "Cohere Rerank" in performance_stats:
            stats = performance_stats["Cohere Rerank"]
            if stats['success']:
                print(f"   • Performance: {stats['avg_time']:.3f}s avg, {stats['classifications_per_second']:.2f} classifications/sec")
            else:
                print(f"   • Performance: Failed - {stats.get('error', 'Unknown error')}")
        
        print("\n📊 Nova Lite Rerank:")
        print("   • Amazon's lightweight LLM, prompt-based approach")
        if "Nova Lite Rerank" in performance_stats:
            stats = performance_stats["Nova Lite Rerank"]
            if stats['success']:
                print(f"   • Performance: {stats['avg_time']:.3f}s avg, {stats['classifications_per_second']:.2f} classifications/sec")
            else:
                print(f"   • Performance: Failed - {stats.get('error', 'Unknown error')}")
        

        
        return 0
        
    except Exception as e:
        print(f"❌ Reranking comparison failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)