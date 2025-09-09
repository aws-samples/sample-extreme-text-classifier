import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Dataset } from '../../types/dataset';

interface GeneratedContentProps {
  dataset: Dataset;
  onDatasetChange: (dataset: Dataset) => void;
  onNext: () => void;
  onBack: () => void;
  isLoading?: boolean;
}

export const GeneratedContent: React.FC<GeneratedContentProps> = ({
  dataset,
  onDatasetChange,
  onNext,
  onBack,
  isLoading = false
}) => {
  const [editingClass, setEditingClass] = useState<string | null>(null);

  const handleClassEdit = (classId: string, field: 'name' | 'description', value: string) => {
    const updatedClasses = dataset.classes.map(cls =>
      cls.id === classId ? { ...cls, [field]: value } : cls
    );
    
    onDatasetChange({
      ...dataset,
      classes: updatedClasses
    });
  };

  const handleExampleEdit = (classId: string, exampleIndex: number, value: string) => {
    const updatedClasses = dataset.classes.map(cls => {
      if (cls.id === classId) {
        const updatedExamples = [...(cls.examples || [])];
        updatedExamples[exampleIndex] = value;
        return { ...cls, examples: updatedExamples };
      }
      return cls;
    });
    
    onDatasetChange({
      ...dataset,
      classes: updatedClasses
    });
  };

  const addExample = (classId: string) => {
    const updatedClasses = dataset.classes.map(cls => {
      if (cls.id === classId) {
        const examples = cls.examples || [];
        return { ...cls, examples: [...examples, 'New example'] };
      }
      return cls;
    });
    
    onDatasetChange({
      ...dataset,
      classes: updatedClasses
    });
  };

  const removeExample = (classId: string, exampleIndex: number) => {
    const updatedClasses = dataset.classes.map(cls => {
      if (cls.id === classId) {
        const examples = cls.examples || [];
        return { ...cls, examples: examples.filter((_, i) => i !== exampleIndex) };
      }
      return cls;
    });
    
    onDatasetChange({
      ...dataset,
      classes: updatedClasses
    });
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Generated Dataset: {dataset.name}</CardTitle>
          <p className="text-sm text-gray-600">{dataset.description}</p>
        </CardHeader>
        <CardContent>
          <div className="bg-green-50 p-4 rounded-lg mb-4">
            <h4 className="font-medium text-green-900 mb-2">✓ Generation Complete!</h4>
            <p className="text-sm text-green-800">
              Generated {dataset.classes.length} classes with sample documents. 
              You can edit anything below before proceeding.
            </p>
          </div>
        </CardContent>
      </Card>

      <div className="space-y-4">
        {dataset.classes.map((cls, index) => (
          <Card key={cls.id} className="border-l-4 border-l-blue-500">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  {editingClass === `${cls.id}-name` ? (
                    <Input
                      value={cls.name}
                      onChange={(e) => handleClassEdit(cls.id, 'name', e.target.value)}
                      onBlur={() => setEditingClass(null)}
                      onKeyDown={(e) => e.key === 'Enter' && setEditingClass(null)}
                      className="font-semibold"
                      autoFocus
                    />
                  ) : (
                    <h3
                      className="text-lg font-semibold cursor-pointer hover:text-blue-600"
                      onClick={() => setEditingClass(`${cls.id}-name`)}
                    >
                      {index + 1}. {cls.name}
                    </h3>
                  )}
                </div>
              </div>
              
              {editingClass === `${cls.id}-description` ? (
                <textarea
                  value={cls.description}
                  onChange={(e) => handleClassEdit(cls.id, 'description', e.target.value)}
                  onBlur={() => setEditingClass(null)}
                  className="w-full p-2 border border-gray-300 rounded text-sm"
                  rows={2}
                  autoFocus
                />
              ) : (
                <p
                  className="text-gray-600 cursor-pointer hover:text-gray-800"
                  onClick={() => setEditingClass(`${cls.id}-description`)}
                >
                  {cls.description}
                </p>
              )}
            </CardHeader>
            
            <CardContent>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-sm">Sample Document</h4>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => addExample(cls.id)}
                  >
                    Add Example
                  </Button>
                </div>
                
                <div className="space-y-2">
                  {(cls.examples || []).map((example, exampleIndex) => (
                    <div key={exampleIndex} className="flex items-start space-x-2">
                      <span className="text-xs text-gray-500 mt-2 w-4">
                        {exampleIndex + 1}.
                      </span>
                      <textarea
                        value={example}
                        onChange={(e) => handleExampleEdit(cls.id, exampleIndex, e.target.value)}
                        className="flex-1 p-2 border border-gray-200 rounded text-sm resize-none"
                        rows={2}
                      />
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => removeExample(cls.id, exampleIndex)}
                        className="text-red-600 hover:text-red-700 mt-1"
                      >
                        ×
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="flex justify-between">
        <Button variant="outline" onClick={onBack}>
          Back
        </Button>
        <Button onClick={onNext} disabled={isLoading}>
          {isLoading ? 'Saving...' : 'Save & Continue'}
        </Button>
      </div>
    </div>
  );
};