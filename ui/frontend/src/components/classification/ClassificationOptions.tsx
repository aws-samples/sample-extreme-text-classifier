/**
 * ClassificationOptions Component - Configuration options for classification
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { ClassificationConfig } from '../../types/classification';
import { useDatasetStore } from '../../stores/dataset-store';
import { attributeService } from '../../services/attribute-service';

interface ClassificationOptionsProps {
  config: ClassificationConfig;
  onConfigChange: (config: ClassificationConfig) => void;
  availableModels?: string[];
  isProcessing?: boolean;
  className?: string;
}

const RERANKING_MODELS = [
  { id: 'amazon_rerank', name: 'Amazon Rerank', description: 'Amazon\'s dedicated reranking model (fastest, most cost-effective)' },
  { id: 'cohere_rerank', name: 'Cohere Rerank v3.5', description: 'Cohere\'s dedicated reranking model' },
  { id: 'us.amazon.nova-lite-v1:0', name: 'Amazon Nova Lite', description: 'Fast and efficient LLM reranking' },
  { id: 'us.amazon.nova-pro-v1:0', name: 'Amazon Nova Pro', description: 'Higher quality LLM reranking' },
  { id: 'us.anthropic.claude-sonnet-4-20250514-v1:0', name: 'Claude Sonnet 4', description: 'Anthropic\'s latest advanced model' },
];

export const ClassificationOptions: React.FC<ClassificationOptionsProps> = ({
  config,
  onConfigChange,
  availableModels = RERANKING_MODELS.map(m => m.id),
  isProcessing = false,
  className = ''
}) => {
  const { selectedDataset } = useDatasetStore();
  const [attributesExist, setAttributesExist] = useState<boolean>(false);
  const [checkingAttributes, setCheckingAttributes] = useState<boolean>(false);

  // Check if attributes exist for the selected dataset
  useEffect(() => {
    const checkAttributes = async () => {
      if (!selectedDataset) {
        setAttributesExist(false);
        // Disable attribute validation if no dataset is selected
        if (config.useAttributeValidation) {
          onConfigChange({
            ...config,
            useAttributeValidation: false
          });
        }
        return;
      }

      setCheckingAttributes(true);
      try {
        const exist = await attributeService.attributesExist(selectedDataset.id);
        setAttributesExist(exist);
        
        // Automatically disable attribute validation if attributes don't exist
        if (config.useAttributeValidation && !exist) {
          onConfigChange({
            ...config,
            useAttributeValidation: false
          });
        }
      } catch (error) {
        console.error('Failed to check attributes:', error);
        setAttributesExist(false);
        // Disable attribute validation on error
        if (config.useAttributeValidation) {
          onConfigChange({
            ...config,
            useAttributeValidation: false
          });
        }
      } finally {
        setCheckingAttributes(false);
      }
    };

    checkAttributes();
  }, [selectedDataset, config, onConfigChange]);

  const handleToggleReranking = () => {
    onConfigChange({
      ...config,
      useReranking: !config.useReranking,
      rerankingModel: !config.useReranking ? availableModels[0] : config.rerankingModel
    });
  };

  const handleToggleAttributeValidation = () => {
    // Only allow enabling if attributes exist
    if (!config.useAttributeValidation && !attributesExist) {
      return; // Don't toggle if trying to enable without attributes
    }
    
    onConfigChange({
      ...config,
      useAttributeValidation: !config.useAttributeValidation
    });
  };



  const handleRerankingModelChange = (modelId: string) => {
    onConfigChange({
      ...config,
      rerankingModel: modelId
    });
  };

  const handleTopKChange = (value: number) => {
    onConfigChange({
      ...config,
      topKCandidates: Math.max(1, Math.min(20, value))
    });
  };

  const selectedModel = RERANKING_MODELS.find(m => m.id === config.rerankingModel);

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Classification Options</CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* Basic Configuration */}
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Top K Candidates
            </label>
            <div className="flex items-center space-x-3">
              <input
                type="range"
                min="1"
                max="20"
                value={config.topKCandidates}
                onChange={(e) => handleTopKChange(parseInt(e.target.value))}
                disabled={isProcessing}
                className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <span className="text-sm font-medium text-gray-900 min-w-[2rem]">
                {config.topKCandidates}
              </span>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Number of top candidates to consider for classification
            </p>
          </div>
        </div>

        {/* Reranking Options */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-medium text-gray-900">LLM Reranking</h4>
              <p className="text-xs text-gray-500">
                Use LLM to rerank similarity search results
              </p>
            </div>
            <Button
              variant={config.useReranking ? 'default' : 'outline'}
              size="sm"
              onClick={handleToggleReranking}
              disabled={isProcessing}
            >
              {config.useReranking ? 'Enabled' : 'Disabled'}
            </Button>
          </div>

          {config.useReranking && (
            <div className="ml-4 space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Reranking Model
                </label>
                <div className="space-y-2">
                  {RERANKING_MODELS.filter(model => availableModels.includes(model.id)).map((model) => (
                    <div
                      key={model.id}
                      className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                        config.rerankingModel === model.id
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 hover:border-gray-300'
                      } ${isProcessing ? 'opacity-50 pointer-events-none' : ''}`}
                      onClick={() => handleRerankingModelChange(model.id)}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-gray-900">{model.name}</p>
                          <p className="text-xs text-gray-500">{model.description}</p>
                        </div>
                        {config.rerankingModel === model.id && (
                          <Badge variant="default">Selected</Badge>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Attribute Validation Options */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-medium text-gray-900">Attribute Validation</h4>
              <p className="text-xs text-gray-500">
                Validate results against class-specific attributes
              </p>
            </div>
            <Button
              variant={config.useAttributeValidation ? 'default' : 'outline'}
              size="sm"
              onClick={handleToggleAttributeValidation}
              disabled={isProcessing || checkingAttributes || (!attributesExist && !config.useAttributeValidation)}
              title={!attributesExist && !config.useAttributeValidation ? 'No attributes available. Please generate attributes first.' : ''}
            >
              {checkingAttributes ? 'Checking...' : config.useAttributeValidation ? 'Enabled' : 'Disabled'}
            </Button>
          </div>

          {config.useAttributeValidation && !attributesExist && (
            <div className="ml-4 p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-800">
                <strong>No attributes available.</strong> Please generate attributes first before enabling attribute validation.
              </p>
            </div>
          )}

          {!config.useAttributeValidation && !attributesExist && selectedDataset && (
            <div className="ml-4 p-3 bg-gray-50 border border-gray-200 rounded-lg">
              <p className="text-sm text-gray-600">
                No attributes available. Please generate attributes first to enable attribute validation.
              </p>
            </div>
          )}
        </div>



        {/* Configuration Summary */}
        <div className="pt-4 border-t border-gray-200">
          <h4 className="text-sm font-medium text-gray-900 mb-3">Current Configuration</h4>
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">
              Top-{config.topKCandidates} candidates
            </Badge>
            {config.useReranking && (
              <Badge variant="default">
                Reranking: {selectedModel?.name || 'Unknown'}
              </Badge>
            )}
            {config.useAttributeValidation && (
              <Badge variant="default">
                Attribute Validation
              </Badge>
            )}
            {!config.useReranking && !config.useAttributeValidation && (
              <Badge variant="secondary">
                Basic Classification Only
              </Badge>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};