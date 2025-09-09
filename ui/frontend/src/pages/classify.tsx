import React, { useState, useCallback, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { DocumentInput, ClassificationOptions, ExampleSelector, AttributeEvaluationTree } from '../components/classification';
import { useDatasetStore } from '../stores/dataset-store';
import { useClassificationStore } from '../stores/classification-store';
import { ClassificationConfig } from '../types/classification';

export const Classify: React.FC = () => {
  const { datasetId } = useParams<{ datasetId?: string }>();
  const { datasets, selectedDataset, setSelectedDataset, loadAllDatasets } = useDatasetStore();
  const { 
    config, 
    updateConfig, 
    isClassifying, 
    currentResult, 
    error,
    pdfExtractionResult,
    classifyText,
    classifyPDF,
    clearError,
    clearPdfExtractionResult
  } = useClassificationStore();

  const [documentText, setDocumentText] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [extractedText, setExtractedText] = useState('');
  const [isExampleSelectorOpen, setIsExampleSelectorOpen] = useState(false);

  // Utility function to format scores with appropriate precision
  const formatScore = (score: number): string => {
    const percentage = score * 100;
    if (percentage < 1) {
      // For scores less than 1%, show 2 significant digits
      return percentage.toPrecision(2) + '%';
    } else {
      // For scores 1% and above, round to nearest integer
      return Math.round(percentage) + '%';
    }
  };

  // Load datasets on component mount
  useEffect(() => {
    if (datasets.length === 0) {
      loadAllDatasets();
    }
  }, [datasets.length, loadAllDatasets]);

  // Auto-select dataset based on URL parameter or first available
  useEffect(() => {
    if (datasets.length > 0) {
      if (datasetId) {
        // Try to find the dataset by ID from URL parameter
        const targetDataset = datasets.find(d => d.id === datasetId);
        if (targetDataset && targetDataset !== selectedDataset) {
          setSelectedDataset(targetDataset);
        }
      } else if (!selectedDataset) {
        // No URL parameter, select first dataset if none selected
        setSelectedDataset(datasets[0]);
      }
    }
  }, [datasetId, selectedDataset, datasets, setSelectedDataset]);

  const handleTextInput = useCallback((text: string) => {
    setDocumentText(text);
    // Clear PDF-related state when switching to text input
    if (selectedFile || pdfExtractionResult) {
      setSelectedFile(null);
      setExtractedText('');
      clearPdfExtractionResult();
    }
    clearError();
  }, [clearError, clearPdfExtractionResult, selectedFile, pdfExtractionResult]);

  const handlePDFUpload = useCallback((file: File) => {
    setSelectedFile(file);
    // Clear previous text content and extracted text when uploading new PDF
    setDocumentText('');
    setExtractedText('');
    clearPdfExtractionResult();
    clearError();
  }, [clearError, clearPdfExtractionResult]);

  const handleExtractedTextChange = useCallback((text: string) => {
    setExtractedText(text);
  }, []);

  const handleConfigChange = useCallback((newConfig: ClassificationConfig) => {
    updateConfig(newConfig);
  }, [updateConfig]);

  const handleDatasetChange = useCallback((datasetId: string) => {
    const dataset = datasets.find(d => d.id === datasetId);
    if (dataset) {
      setSelectedDataset(dataset);
    }
  }, [datasets, setSelectedDataset]);

  const handleClassifyText = useCallback(async () => {
    if (!selectedDataset || !documentText.trim()) {
      return;
    }

    try {
      await classifyText({
        text: documentText,
        datasetId: selectedDataset.id,
        config
      });
    } catch (error) {
      console.error('Classification failed:', error);
    }
  }, [selectedDataset, documentText, config, classifyText]);

  const handleClassifyPDF = useCallback(async () => {
    if (!selectedDataset || !selectedFile) {
      return;
    }

    try {
      await classifyPDF({
        file: selectedFile,
        datasetId: selectedDataset.id,
        config
      });
    } catch (error) {
      console.error('PDF classification failed:', error);
    }
  }, [selectedDataset, selectedFile, config, classifyPDF]);

  const handleLoadExample = useCallback(() => {
    setIsExampleSelectorOpen(true);
  }, []);

  const handleSelectExample = useCallback((text: string) => {
    setDocumentText(text);
    handleTextInput(text);
    setIsExampleSelectorOpen(false);
  }, [handleTextInput]);

  const canClassify = Boolean(selectedDataset && (documentText.trim() || selectedFile));
  const hasEmbeddings = selectedDataset?.embeddingsGenerated;

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Document Classification</h1>
        <p className="text-gray-600 mt-1">Classify text and PDF documents with multiple approaches</p>
      </div>

      {/* Dataset Selection */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Dataset Selection</CardTitle>
        </CardHeader>
        <CardContent>
          {datasets.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-gray-500 mb-4">No datasets available</p>
              <p className="text-sm text-gray-400">
                Please create a dataset in the Datasets section before classifying documents.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Dataset
                </label>
                <select
                  value={selectedDataset?.id || ''}
                  onChange={(e) => handleDatasetChange(e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  disabled={isClassifying}
                >
                  <option value="">Select a dataset...</option>
                  {datasets.map((dataset) => (
                    <option key={dataset.id} value={dataset.id}>
                      {dataset.name} ({dataset.classes.length} classes)
                    </option>
                  ))}
                </select>
              </div>

              {selectedDataset && (
                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium text-gray-900">{selectedDataset.name}</h4>
                    <div className="flex space-x-2">
                      <Badge variant="secondary">
                        {selectedDataset.classes.length} classes
                      </Badge>
                      <Badge variant={hasEmbeddings ? 'success' : 'warning'}>
                        {hasEmbeddings ? 'Embeddings Ready' : 'No Embeddings'}
                      </Badge>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600">{selectedDataset.description}</p>
                  
                  {!hasEmbeddings && (
                    <div className="mt-3 p-3 bg-yellow-50 border border-yellow-200 rounded-md">
                      <p className="text-sm text-yellow-800">
                        <strong>Note:</strong> This dataset doesn't have embeddings generated yet. 
                        Embeddings will be generated automatically during classification, which may take longer.
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {selectedDataset && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column - Input and Options */}
          <div className="space-y-6">
            {/* Document Input */}
            <DocumentInput
              onTextInput={handleTextInput}
              onPDFUpload={handlePDFUpload}
              onExtractedTextChange={handleExtractedTextChange}
              onClassify={documentText ? handleClassifyText : handleClassifyPDF}
              onLoadExample={handleLoadExample}
              isProcessing={isClassifying}
              canClassify={canClassify}
              classifyButtonText={`Classify ${selectedFile ? 'PDF' : 'Text'}`}
              extractedText={extractedText}
              pdfExtractionResult={pdfExtractionResult || undefined}
              dataset={selectedDataset}
              textValue={documentText}
            />

            {/* Classification Options */}
            <ClassificationOptions
              config={config}
              onConfigChange={handleConfigChange}
              isProcessing={isClassifying}
            />


          </div>

          {/* Right Column - Results */}
          <div className="space-y-6">
            {/* Error Display */}
            {error && (
              <Card className="border-red-200">
                <CardContent className="pt-6">
                  <div className="flex items-start space-x-3">
                    <div className="flex-shrink-0">
                      <svg className="h-5 w-5 text-red-500" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div>
                      <h4 className="text-sm font-medium text-red-800">Classification Error</h4>
                      <p className="text-sm text-red-700 mt-1">{error.message}</p>
                      {error.details && (
                        <p className="text-xs text-red-600 mt-1">{error.details}</p>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Results Display */}
            {currentResult ? (
              <Card>
                <CardHeader>
                  <CardTitle>Classification Results</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {/* Primary Result */}
                    <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium text-green-900">
                          {currentResult.predictedClass.name}
                        </h4>
                      </div>
                      <p className="text-sm text-green-800">
                        {currentResult.predictedClass.description}
                      </p>
                    </div>

                    {/* Score */}
                    <div className="space-y-3">
                      <h5 className="font-medium text-gray-900">Score</h5>
                      
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Similarity Score</span>
                          <Badge variant="secondary">
                            {Math.round(currentResult.similarityScore * 100)}%
                          </Badge>
                        </div>
                        
                        {currentResult.rerankScore !== undefined && currentResult.rerankScore !== null && (
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-gray-600">Rerank Score</span>
                            <Badge variant="secondary">
                              {formatScore(currentResult.rerankScore)}
                            </Badge>
                          </div>
                        )}
                        
                        {currentResult.attributeScore !== undefined && currentResult.attributeScore !== null && (
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-gray-600">Attribute Score</span>
                            <Badge variant="secondary">
                              {Math.round(currentResult.attributeScore * 100)}%
                            </Badge>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Top K Candidates */}
                    {currentResult.alternatives.length > 0 && (
                      <div className="space-y-3">
                        <h5 className="font-medium text-gray-900">Top K Candidates</h5>
                        
                        {/* Check if reranking was used */}
                        {config.useReranking && currentResult.alternatives.some(alt => alt.scores.rerank !== undefined && alt.scores.rerank !== null) ? (
                          /* Two-column layout for reranking results */
                          <div className="grid grid-cols-2 gap-4">
                            {/* Similarity Search Results Column */}
                            <div className="space-y-2">
                              <h6 className="text-sm font-medium text-gray-700 border-b pb-1">Similarity Search</h6>
                              {currentResult.alternatives
                                .slice()
                                .sort((a, b) => b.scores.similarity - a.scores.similarity)
                                .slice(0, 3)
                                .map((alt, index) => (
                                  <div key={`sim-${index}`} className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                                    <div className="flex items-center justify-between">
                                      <span className="text-sm font-medium text-gray-900">
                                        {alt.class.name}
                                      </span>
                                      <Badge variant="secondary">
                                        {Math.round(alt.scores.similarity * 100)}%
                                      </Badge>
                                    </div>
                                    <p className="text-xs text-gray-600 mt-1">
                                      {alt.class.description}
                                    </p>
                                  </div>
                                ))}
                            </div>

                            {/* Reranked Results Column */}
                            <div className="space-y-2">
                              <h6 className="text-sm font-medium text-gray-700 border-b pb-1">After Rerank</h6>
                              {currentResult.alternatives
                                .slice()
                                .sort((a, b) => (b.scores.rerank || 0) - (a.scores.rerank || 0))
                                .slice(0, 3)
                                .map((alt, index) => (
                                  <div key={`rerank-${index}`} className="p-3 bg-green-50 rounded-lg border border-green-200">
                                    <div className="flex items-center justify-between">
                                      <span className="text-sm font-medium text-gray-900">
                                        {alt.class.name}
                                      </span>
                                      <Badge variant="secondary">
                                        {formatScore(alt.scores.rerank || 0)}
                                      </Badge>
                                    </div>
                                    <p className="text-xs text-gray-600 mt-1">
                                      {alt.class.description}
                                    </p>
                                  </div>
                                ))}
                            </div>
                          </div>
                        ) : (
                          /* Single column layout for non-reranking results */
                          <div className="space-y-2">
                            {currentResult.alternatives.slice(0, 3).map((alt, index) => (
                              <div key={index} className="p-3 bg-gray-50 rounded-lg">
                                <div className="flex items-center justify-between">
                                  <span className="text-sm font-medium text-gray-900">
                                    {alt.class.name}
                                  </span>
                                  <Badge variant="secondary">
                                    {Math.round(alt.confidence * 100)}%
                                  </Badge>
                                </div>
                                <p className="text-xs text-gray-600 mt-1">
                                  {alt.class.description}
                                </p>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Attribute Validation Results */}
                    {currentResult.attributeValidation && (
                      <div className="space-y-3">
                        <h5 className="font-medium text-gray-900">Attribute Validation</h5>
                        
                        {/* Show evaluation tree if available, otherwise fall back to flat list */}
                        {currentResult.attributeValidation.evaluationTree ? (
                          <div className="space-y-2">
                            <p className="text-sm text-gray-600 mb-3">
                              Logical structure showing how conditions were evaluated:
                            </p>
                            <AttributeEvaluationTree tree={currentResult.attributeValidation.evaluationTree} />
                          </div>
                        ) : (
                          /* Fallback to flat list display */
                          <div className="space-y-3">
                            {/* Conditions Met */}
                            {currentResult.attributeValidation.conditionsMet.length > 0 && (
                              <div className="space-y-2">
                                <h6 className="text-sm font-medium text-green-700">Conditions met</h6>
                                <div className="space-y-1">
                                  {currentResult.attributeValidation.conditionsMet.map((condition, index) => (
                                    <div key={index} className="flex items-start space-x-2">
                                      <svg className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                      </svg>
                                      <span className="text-sm text-green-700">{condition}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}

                            {/* Conditions Not Met */}
                            {currentResult.attributeValidation.conditionsNotMet.length > 0 && (
                              <div className="space-y-2">
                                <h6 className="text-sm font-medium text-red-700">Conditions not met</h6>
                                <div className="space-y-1">
                                  {currentResult.attributeValidation.conditionsNotMet.map((condition, index) => (
                                    <div key={index} className="flex items-start space-x-2">
                                      <svg className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                                        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                                      </svg>
                                      <span className="text-sm text-red-700">{condition}</span>
                                    </div>
                                  ))}
                                </div>
                                
                                {/* Explanatory note if conditions are not met but attribute score is 100% */}
                                {currentResult.attributeScore === 1.0 && (
                                  <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                                    <div className="flex items-start space-x-2">
                                      <svg className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                                      </svg>
                                      <div>
                                        <p className="text-sm text-blue-800 font-medium">Note about 100% attribute score</p>
                                        <p className="text-sm text-blue-700 mt-1">
                                          Some conditions are not met, but the attribute score is 100%. This can occur when conditions are part of an OR group where only one condition needs to be satisfied.
                                        </p>
                                      </div>
                                    </div>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Processing Time */}
                    {currentResult.processingTime && (
                      <div className="text-xs text-gray-500 text-center pt-2 border-t">
                        Processed in {currentResult.processingTime}ms
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            ) : (
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center py-8">
                    <div className="text-gray-400 mb-4">
                      <svg className="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                      </svg>
                    </div>
                    <p className="text-gray-500">
                      Classification results will appear here
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      )}

      {/* Example Selector Modal */}
      <ExampleSelector
        isOpen={isExampleSelectorOpen}
        onClose={() => setIsExampleSelectorOpen(false)}
        onSelectExample={handleSelectExample}
        dataset={selectedDataset}
      />
    </div>
  );
};