import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Dataset } from '../../types/dataset';

interface WizardCompleteProps {
  dataset: Dataset;
  onGoToDatasets: () => void;
  onGoToAttributes: () => void;
  onGoToClassify: () => void;
  onStartOver: () => void;
}

export const WizardComplete: React.FC<WizardCompleteProps> = ({
  dataset,
  onGoToDatasets,
  onGoToAttributes,
  onGoToClassify,
  onStartOver
}) => {
  return (
    <div className="space-y-6">
      <Card className="border-green-200 bg-green-50">
        <CardHeader>
          <CardTitle className="text-green-800 flex items-center">
            <span className="mr-2">🎉</span>
            Domain Setup Complete!
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold text-green-900">Dataset Created: {dataset.name}</h3>
              <p className="text-green-700 text-sm">{dataset.description}</p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
              <div className="bg-white p-4 rounded-lg border">
                <div className="text-2xl font-bold text-blue-600">{dataset.classes.length}</div>
                <div className="text-sm text-gray-600">Classes Generated</div>
              </div>
              
              <div className="bg-white p-4 rounded-lg border">
                <div className="text-2xl font-bold text-green-600">
                  {dataset.classes.reduce((total, cls) => total + (cls.examples?.length || 0), 0)}
                </div>
                <div className="text-sm text-gray-600">Sample Documents</div>
              </div>
              
              <div className="bg-white p-4 rounded-lg border">
                <div className="text-2xl font-bold text-purple-600">Ready</div>
                <div className="text-sm text-gray-600">For Classification</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>What's Next?</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <Button
                onClick={onGoToClassify}
                className="justify-start h-auto p-3"
              >
                <div className="text-left">
                  <div className="font-semibold">🚀 Start Classifying</div>
                  <div className="text-xs text-white/90">
                    Test your new dataset
                  </div>
                </div>
              </Button>
              
              <Button
                onClick={onGoToAttributes}
                className="justify-start h-auto p-3"
                variant="outline"
              >
                <div className="text-left">
                  <div className="font-semibold">🎯 Generate Attributes (Optional)</div>
                  <div className="text-xs text-gray-600">
                    Improve classification accuracy
                  </div>
                </div>
              </Button>
            </div>
            
            <div className="pt-4 border-t border-gray-200">
              <div className="flex justify-between">
                <Button
                  onClick={onGoToDatasets}
                  variant="ghost"
                  size="sm"
                >
                  Review Dataset
                </Button>
                
                <Button
                  onClick={onStartOver}
                  variant="ghost"
                  size="sm"
                >
                  Create Another
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="bg-blue-50 border-blue-200">
        <CardContent className="p-4">
          <h4 className="font-semibold text-blue-900 mb-2">💡 Pro Tips</h4>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• Generate attributes to improve classification accuracy with validation rules</li>
            <li>• Try different reranking models in the classify section for better results</li>
            <li>• Upload PDF documents to test multimodal classification capabilities</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
};