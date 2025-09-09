/**
 * ExampleSelector Component - Modal for selecting examples from dataset classes
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Dataset, ClassDefinition } from '../../types/dataset';

interface ExampleItem {
  text: string;
  className: string;
  classDescription: string;
}

interface ExampleSelectorProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectExample: (text: string) => void;
  dataset: Dataset | null;
}

export const ExampleSelector: React.FC<ExampleSelectorProps> = ({
  isOpen,
  onClose,
  onSelectExample,
  dataset
}) => {
  const [examples, setExamples] = useState<ExampleItem[]>([]);
  const [selectedClassFilter, setSelectedClassFilter] = useState<string>('all');

  useEffect(() => {
    if (dataset && isOpen) {
      // Collect all examples from all classes
      const allExamples: ExampleItem[] = [];
      
      dataset.classes.forEach(classDefinition => {
        if (classDefinition.examples && classDefinition.examples.length > 0) {
          classDefinition.examples.forEach(example => {
            allExamples.push({
              text: example,
              className: classDefinition.name,
              classDescription: classDefinition.description
            });
          });
        }
      });
      
      setExamples(allExamples);
    }
  }, [dataset, isOpen]);

  const handleSelectExample = (example: ExampleItem) => {
    onSelectExample(example.text);
    onClose();
  };

  const filteredExamples = selectedClassFilter === 'all' 
    ? examples 
    : examples.filter(example => example.className === selectedClassFilter);

  const availableClasses = dataset?.classes.filter(cls => cls.examples && cls.examples.length > 0) || [];

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Load Example</h2>
            <p className="text-sm text-gray-600 mt-1">
              Select an example from {dataset?.name || 'the dataset'} to populate the text field
            </p>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </Button>
        </div>

        {/* Filter */}
        {availableClasses.length > 1 && (
          <div className="p-4 border-b bg-gray-50">
            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium text-gray-700">Filter by class:</label>
              <select
                value={selectedClassFilter}
                onChange={(e) => setSelectedClassFilter(e.target.value)}
                className="text-sm border border-gray-300 rounded-md px-2 py-1 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="all">All classes ({examples.length} examples)</option>
                {availableClasses.map(cls => {
                  const count = cls.examples?.length || 0;
                  return (
                    <option key={cls.name} value={cls.name}>
                      {cls.name} ({count} examples)
                    </option>
                  );
                })}
              </select>
            </div>
          </div>
        )}

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {examples.length === 0 ? (
            <div className="text-center py-12">
              <div className="text-gray-400 mb-4">
                <svg className="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <p className="text-gray-500 text-lg">No examples available</p>
              <p className="text-gray-400 text-sm mt-2">
                This dataset doesn't contain any examples yet. You can add examples when editing the dataset.
              </p>
            </div>
          ) : filteredExamples.length === 0 ? (
            <div className="text-center py-12">
              <div className="text-gray-400 mb-4">
                <svg className="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <p className="text-gray-500 text-lg">No examples found</p>
              <p className="text-gray-400 text-sm mt-2">
                No examples available for the selected class filter.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {filteredExamples.map((example, index) => (
                <Card key={index} className="hover:shadow-md transition-shadow cursor-pointer" onClick={() => handleSelectExample(example)}>
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-sm font-medium text-gray-900">
                        {example.className}
                      </CardTitle>
                      <Badge variant="secondary" className="text-xs">
                        Ground Truth
                      </Badge>
                    </div>
                    <p className="text-xs text-gray-600 mt-1">
                      {example.classDescription}
                    </p>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <div className="bg-gray-50 rounded-md p-3">
                      <p className="text-sm text-gray-800 leading-relaxed">
                        {example.text.length > 300 
                          ? `${example.text.substring(0, 300)}...` 
                          : example.text
                        }
                      </p>
                      {example.text.length > 300 && (
                        <p className="text-xs text-gray-500 mt-2">
                          Click to load full text ({example.text.length} characters)
                        </p>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-4 border-t bg-gray-50">
          <div className="text-sm text-gray-600">
            {filteredExamples.length > 0 && (
              <>Showing {filteredExamples.length} example{filteredExamples.length !== 1 ? 's' : ''}</>
            )}
          </div>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
        </div>
      </div>
    </div>
  );
};