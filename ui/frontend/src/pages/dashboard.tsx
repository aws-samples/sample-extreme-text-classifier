import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { useDatasetStore } from '../stores/dataset-store';

export const Dashboard: React.FC = () => {
  const { datasets, loadAllDatasets } = useDatasetStore();

  useEffect(() => {
    // Load datasets on component mount to get statistics
    loadAllDatasets();
  }, [loadAllDatasets]);

  const totalClasses = datasets.reduce((sum, dataset) => sum + dataset.classes.length, 0);
  const datasetsWithEmbeddings = datasets.filter(d => d.embeddingsGenerated).length;

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Text Classifier Showcase
        </h1>
        <p className="text-lg text-gray-600">
          Explore the capabilities of the multi-class text classifier library through this interactive demo.
        </p>
      </div>

      {/* Statistics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Datasets</p>
                <p className="text-2xl font-bold text-gray-900">{datasets.length}</p>
              </div>
              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Classes</p>
                <p className="text-2xl font-bold text-gray-900">{totalClasses}</p>
              </div>
              <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                <svg className="w-4 h-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
                </svg>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Ready Datasets</p>
                <p className="text-2xl font-bold text-gray-900">{datasetsWithEmbeddings}</p>
              </div>
              <div className="w-8 h-8 bg-yellow-100 rounded-full flex items-center justify-center">
                <svg className="w-4 h-4 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Available Models</p>
                <p className="text-2xl font-bold text-gray-900">5+</p>
              </div>
              <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center">
                <svg className="w-4 h-4 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Classification Workflow */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle className="text-xl">How Classification Works</CardTitle>
          <CardDescription>
            A 3-step pipeline that combines semantic understanding with optional quality enhancements
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col lg:flex-row items-center justify-center space-y-6 lg:space-y-0 lg:space-x-8 py-6">
            {/* Step 1 */}
            <div className="flex flex-col items-center text-center max-w-xs">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-4">
                <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <h3 className="font-semibold text-lg mb-2">🔍 Semantic Search</h3>
              <p className="text-sm text-gray-600 mb-2">Find similar classes from 1000+ options using embedding-based similarity</p>
              <Badge variant="default">Required</Badge>
            </div>

            {/* Arrow */}
            <div className="hidden lg:block">
              <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </div>

            {/* Step 2 */}
            <div className="flex flex-col items-center text-center max-w-xs">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mb-4">
                <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="font-semibold text-lg mb-2">🎯 LLM Reranking</h3>
              <p className="text-sm text-gray-600 mb-2">Refine top candidates using Amazon Bedrock for deeper understanding</p>
              <Badge variant="secondary">Optional</Badge>
            </div>

            {/* Arrow */}
            <div className="hidden lg:block">
              <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </div>

            {/* Step 3 */}
            <div className="flex flex-col items-center text-center max-w-xs">
              <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mb-4">
                <svg className="w-8 h-8 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <h3 className="font-semibold text-lg mb-2">✅ Attribute Validation</h3>
              <p className="text-sm text-gray-600 mb-2">Validate predictions against business rules and conditions</p>
              <Badge variant="secondary">Optional</Badge>
            </div>
          </div>

          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-700">
              <strong>Example:</strong> If classifying as "Invoice", the system validates it has a monetary amount, 
              identifies both parties, and contains either a payment request or proof of payment.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Get Started Section */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle className="text-xl">Get Started</CardTitle>
          <CardDescription>
            Choose your path to explore the multi-class text classifier
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Quick Classification */}
            <div className="p-6 border rounded-lg hover:shadow-md transition-shadow">
              <div className="flex items-center space-x-3 mb-4">
                <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                  <span className="text-blue-600 font-bold text-lg">1</span>
                </div>
                <h3 className="text-lg font-semibold">Classify Text</h3>
              </div>
              <p className="text-gray-600 mb-4">
                Start by classifying text or PDF documents using our pre-built datasets. 
                Try different approaches: basic similarity, LLM reranking, or attribute validation.
              </p>
              <Link to="/classify">
                <Button className="w-full">
                  Start Classifying
                </Button>
              </Link>
            </div>

            {/* Advanced Setup */}
            <div className="p-6 border rounded-lg hover:shadow-md transition-shadow">
              <div className="flex items-center space-x-3 mb-4">
                <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                  <span className="text-green-600 font-bold text-lg">2</span>
                </div>
                <h3 className="text-lg font-semibold">Build Your Domain</h3>
              </div>
              <p className="text-gray-600 mb-4">
                Create classification datasets for your specific domain or generate attributes 
                to enhance classification accuracy with business rules.
              </p>
              <div className="flex space-x-2">
                <Link to="/wizard" className="flex-1">
                  <Button variant="outline" className="w-full">
                    Domain Wizard
                  </Button>
                </Link>
                <Link to="/attributes" className="flex-1">
                  <Button variant="outline" className="w-full">
                    Generate Attributes
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>


    </div>
  );
};