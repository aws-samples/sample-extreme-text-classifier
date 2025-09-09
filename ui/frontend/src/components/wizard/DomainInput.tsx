import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Input } from '../ui/input';
import { Button } from '../ui/button';
import { Slider } from '../ui/slider';

interface DomainInputProps {
  domain: string;
  classCount: number;
  onDomainChange: (domain: string) => void;
  onClassCountChange: (count: number) => void;
  onNext: () => void;
  isLoading?: boolean;
}

export const DomainInput: React.FC<DomainInputProps> = ({
  domain,
  classCount,
  onDomainChange,
  onClassCountChange,
  onNext,
  isLoading = false
}) => {

  const isValid = domain.trim().length >= 3 && classCount >= 2 && classCount <= 10;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Domain Setup</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Domain Description
          </label>
          <Input
            type="text"
            value={domain}
            onChange={(e) => onDomainChange(e.target.value)}
            placeholder="e.g., medical supplies, legal documents, customer support tickets"
            className="w-full"
          />
          <p className="text-sm text-gray-500 mt-1">
            Describe the type of documents you want to classify
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Number of Classes: {classCount}
          </label>
          <div className="max-w-xs">
            <div className="px-2">
              <Slider
                value={[classCount]}
                onValueChange={(value) => onClassCountChange(value[0])}
                min={2}
                max={10}
                step={1}
                className="w-full"
              />
            </div>
            <div className="flex justify-between text-xs text-gray-500 mt-1 px-2">
              <span>2</span>
              <span>10</span>
            </div>
          </div>
          <p className="text-sm text-gray-500 mt-1">
            How many different categories do you want?
          </p>
        </div>

        <div className="bg-blue-50 p-4 rounded-lg">
          <h4 className="font-medium text-blue-900 mb-2">What happens next?</h4>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• AI will generate {classCount} relevant classes for your domain</li>
            <li>• One sample document will be created for each class</li>
            <li>• You can review and edit everything before using</li>
          </ul>
        </div>

        <div className="flex justify-end">
          <Button
            onClick={onNext}
            disabled={!isValid || isLoading}
            className="px-6"
          >
            {isLoading ? 'Generating...' : 'Generate Classes'}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};