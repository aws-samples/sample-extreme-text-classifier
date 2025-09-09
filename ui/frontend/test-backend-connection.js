/**
 * Simple test script to verify backend connection
 * Run this with: node test-backend-connection.js
 */

const API_BASE_URL = 'http://localhost:8000';

async function testBackendConnection() {
  console.log('🔍 Testing backend connection...\n');

  try {
    // Test 1: Health check
    console.log('1. Testing health endpoint...');
    const healthResponse = await fetch(`${API_BASE_URL}/health`);
    const healthData = await healthResponse.json();
    console.log('✅ Health check:', healthData.status);
    console.log('   Mode:', healthData.mode);
    console.log('   Classifier library:', healthData.classifier_library);

    // Test 2: List datasets
    console.log('\n2. Testing datasets endpoint...');
    const datasetsResponse = await fetch(`${API_BASE_URL}/datasets`);
    const datasets = await datasetsResponse.json();
    console.log('✅ Datasets found:', datasets.length);
    datasets.forEach(dataset => {
      console.log(`   - ${dataset.name} (${dataset.classes.length} classes)`);
    });

    // Test 3: Test classification with sample dataset
    if (datasets.length > 0) {
      console.log('\n3. Testing text classification...');
      const testDataset = datasets[0];
      
      const classificationRequest = {
        text: "Please remit payment of $1,250.00 within 30 days for services rendered.",
        dataset_id: testDataset.id,
        config: {
          use_reranking: false,
          use_attribute_validation: false,
          top_k_candidates: 5
        }
      };

      const classifyResponse = await fetch(`${API_BASE_URL}/classify/text`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(classificationRequest)
      });

      if (classifyResponse.ok) {
        const result = await classifyResponse.json();
        console.log('✅ Classification successful!');
        console.log(`   Predicted: ${result.predicted_class.name}`);
        console.log(`   Confidence: ${(result.effective_score * 100).toFixed(1)}%`);
        console.log(`   Similarity Score: ${(result.similarity_score * 100).toFixed(1)}%`);
      } else {
        const error = await classifyResponse.json();
        console.log('❌ Classification failed:', error.error);
      }
    }

    console.log('\n🎉 Backend connection test completed successfully!');
    console.log('\n📝 Next steps:');
    console.log('   1. Start the backend: cd ui/backend && python start_backend.py');
    console.log('   2. Start the frontend: cd ui/frontend && npm start');
    console.log('   3. Open http://localhost:3000 and test the classification interface');

  } catch (error) {
    console.log('❌ Backend connection failed:', error.message);
    console.log('\n🔧 Troubleshooting:');
    console.log('   1. Make sure the backend is running: cd ui/backend && python start_backend.py');
    console.log('   2. Check that port 8000 is available');
    console.log('   3. Verify the backend URL is correct:', API_BASE_URL);
  }
}

// Run the test
testBackendConnection();