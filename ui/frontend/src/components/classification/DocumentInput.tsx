/**
 * DocumentInput Component - Tabbed interface for text input vs PDF upload
 */

import React, { useState, useCallback, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Input, Textarea } from '../ui/input';
import { Badge } from '../ui/badge';
import { Dataset } from '../../types/dataset';

interface DocumentInputProps {
  onTextInput: (text: string) => void;
  onPDFUpload: (file: File) => void;
  onExtractedTextChange?: (text: string) => void;
  onClassify?: () => void;
  onLoadExample?: () => void;
  isProcessing?: boolean;
  canClassify?: boolean;
  classifyButtonText?: string;
  extractedText?: string;
  pdfExtractionResult?: {
    extractedText: string;
    pageCount: number;
    extractionMethod: string;
    confidence: number;
  };
  className?: string;
  dataset?: Dataset | null;
  textValue?: string; // Add controlled text value prop
}

export const DocumentInput: React.FC<DocumentInputProps> = ({
  onTextInput,
  onPDFUpload,
  onExtractedTextChange,
  onClassify,
  onLoadExample,
  isProcessing = false,
  canClassify = false,
  classifyButtonText = 'Classify',
  extractedText,
  pdfExtractionResult,
  className = '',
  dataset,
  textValue = ''
}) => {
  const [activeTab, setActiveTab] = useState<'text' | 'pdf'>('text');
  const [textContent, setTextContent] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [fileValidationError, setFileValidationError] = useState<string | null>(null);

  // Sync internal state with controlled value from parent
  useEffect(() => {
    if (textValue !== textContent) {
      setTextContent(textValue);
    }
  }, [textValue, textContent]);

  const handleTextChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const text = e.target.value;
    setTextContent(text);
    onTextInput(text);
  }, [onTextInput]);

  const handleFileSelect = useCallback((file: File) => {
    // Validate file
    if (file.type !== 'application/pdf') {
      setFileValidationError('Only PDF files are supported');
      return;
    }

    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
      setFileValidationError('File size must be less than 50MB');
      return;
    }

    if (file.size === 0) {
      setFileValidationError('File cannot be empty');
      return;
    }

    setFileValidationError(null);
    setUploadedFile(file);
    onPDFUpload(file);
  }, [onPDFUpload]);

  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  }, [handleFileSelect]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileSelect(file);
    }
  }, [handleFileSelect]);

  const handleExtractedTextChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const text = e.target.value;
    onExtractedTextChange?.(text);
  }, [onExtractedTextChange]);

  const clearFile = useCallback(() => {
    setUploadedFile(null);
    setFileValidationError(null);
  }, []);

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Document Input</CardTitle>
        <div className="flex space-x-1 mt-4">
          <Button
            variant={activeTab === 'text' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setActiveTab('text')}
            disabled={isProcessing}
          >
            Text Input
          </Button>
          <Button
            variant={activeTab === 'pdf' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setActiveTab('pdf')}
            disabled={isProcessing}
          >
            PDF Upload
          </Button>
        </div>
      </CardHeader>
      
      <CardContent>
        {activeTab === 'text' && (
          <div className="space-y-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <label htmlFor="text-input" className="block text-sm font-medium text-gray-700">
                  Document Text
                </label>
                {onLoadExample && dataset && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={onLoadExample}
                    disabled={isProcessing}
                    className="text-xs"
                  >
                    <svg className="h-3 w-3 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    Load Example
                  </Button>
                )}
              </div>
              <Textarea
                id="text-input"
                placeholder="Enter or paste your document text here..."
                value={textContent}
                onChange={handleTextChange}
                disabled={isProcessing}
                className="min-h-[200px] resize-y"
              />
            </div>
            
            <div className="flex items-center justify-between">
              {textContent && (
                <div className="text-sm text-gray-600">
                  <Badge variant="secondary">
                    {textContent.length} characters
                  </Badge>
                </div>
              )}
              
              {onClassify && (
                <Button
                  onClick={onClassify}
                  disabled={!canClassify || isProcessing}
                  className="ml-auto"
                >
                  {isProcessing ? (
                    <div className="flex items-center space-x-2">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                      <span>Classifying...</span>
                    </div>
                  ) : (
                    classifyButtonText
                  )}
                </Button>
              )}
            </div>
          </div>
        )}

        {activeTab === 'pdf' && (
          <div className="space-y-4">
            {!uploadedFile ? (
              <>
                <div
                  className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${
                    isDragOver 
                      ? 'border-blue-500 bg-blue-50' 
                      : 'border-gray-300 hover:border-gray-400'
                  } ${isProcessing ? 'opacity-50 pointer-events-none' : ''}`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => document.getElementById('pdf-upload')?.click()}
                >
                  <div className="space-y-4">
                    <div className="text-gray-500">
                      <svg className="mx-auto h-12 w-12" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                        <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </div>
                    
                    <div>
                      <p className="text-lg font-medium text-gray-900">
                        Drop your PDF file here
                      </p>
                      <p className="text-sm text-gray-500 mt-1">
                        or click to browse files
                      </p>
                    </div>
                    
                    <div>
                      <Input
                        type="file"
                        accept=".pdf"
                        onChange={handleFileInputChange}
                        disabled={isProcessing}
                        className="hidden"
                        id="pdf-upload"
                      />
                      <label htmlFor="pdf-upload" className="cursor-pointer inline-block">
                        <Button variant="outline" size="sm" disabled={isProcessing} type="button">
                          Browse Files
                        </Button>
                      </label>
                    </div>
                    
                    <p className="text-xs text-gray-400">
                      PDF files up to 50MB
                    </p>
                  </div>
                </div>

                {/* File Validation Error */}
                {fileValidationError && (
                  <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <svg className="h-4 w-4 text-red-500" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                      </svg>
                      <span className="text-sm text-red-700">{fileValidationError}</span>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="space-y-4">
                {/* File Info */}
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="flex-shrink-0">
                      <svg className="h-8 w-8 text-red-500" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-900">{uploadedFile.name}</p>
                      <p className="text-xs text-gray-500">
                        {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                  
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={clearFile}
                    disabled={isProcessing}
                  >
                    Remove
                  </Button>
                </div>

                {/* Classify Button - Show immediately after file upload */}
                {!pdfExtractionResult && onClassify && (
                  <div className="flex justify-end">
                    <Button
                      onClick={onClassify}
                      disabled={!canClassify || isProcessing}
                    >
                      {isProcessing ? (
                        <div className="flex items-center space-x-2">
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                          <span>Processing PDF...</span>
                        </div>
                      ) : (
                        classifyButtonText
                      )}
                    </Button>
                  </div>
                )}

                {/* Processing Status */}
                {isProcessing && (
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                      <span className="text-sm text-blue-700">
                        Extracting content from PDF...
                      </span>
                    </div>
                  </div>
                )}

                {/* Extraction Results */}
                {pdfExtractionResult && (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <h4 className="text-sm font-medium text-gray-900">Extracted Content</h4>
                      <div className="flex space-x-2">
                        <Badge variant="secondary">
                          {pdfExtractionResult.pageCount} pages
                        </Badge>
                        <Badge variant="secondary">
                          {pdfExtractionResult.extractionMethod}
                        </Badge>
                        <Badge 
                          variant={pdfExtractionResult.confidence > 0.8 ? 'success' : 'warning'}
                        >
                          {Math.round(pdfExtractionResult.confidence * 100)}% confidence
                        </Badge>
                      </div>
                    </div>
                    
                    <Textarea
                      value={extractedText || pdfExtractionResult.extractedText}
                      onChange={handleExtractedTextChange}
                      placeholder="Extracted text will appear here..."
                      className="min-h-[200px] resize-y"
                      disabled={isProcessing}
                    />
                    
                    <div className="flex items-center justify-between">
                      <div className="text-sm text-gray-600">
                        <Badge variant="secondary">
                          {(extractedText || pdfExtractionResult.extractedText).length} characters
                        </Badge>
                      </div>
                      
                      {onClassify && (
                        <Button
                          onClick={onClassify}
                          disabled={!canClassify || isProcessing}
                          className="ml-auto"
                        >
                          {isProcessing ? (
                            <div className="flex items-center space-x-2">
                              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                              <span>Classifying...</span>
                            </div>
                          ) : (
                            classifyButtonText
                          )}
                        </Button>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};