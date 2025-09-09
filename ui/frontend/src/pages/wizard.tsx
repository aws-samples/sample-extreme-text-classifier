import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { WizardStepper, DomainInput, GeneratedContent, WizardComplete } from '../components/wizard';
import { wizardService } from '../services/wizard-service';
import { datasetService } from '../services/dataset-service';
import { useDatasetStore } from '../stores/dataset-store';
import { Dataset } from '../types/dataset';
import { ErrorHandler } from '../utils/error-handler';

type WizardStep = 'domain' | 'generate' | 'review' | 'complete';

interface WizardState {
  currentStep: WizardStep;
  domain: string;
  classCount: number;
  dataset: Dataset | null;
  isLoading: boolean;
  error: string | null;
}

export const Wizard: React.FC = () => {
  const navigate = useNavigate();
  const { addDataset } = useDatasetStore();
  
  const [state, setState] = useState<WizardState>({
    currentStep: 'domain',
    domain: '',
    classCount: 5,
    dataset: null,
    isLoading: false,
    error: null
  });

  const steps = [
    {
      id: 'domain',
      title: 'Domain Setup',
      description: 'Specify your domain and preferences',
      completed: state.currentStep !== 'domain',
      current: state.currentStep === 'domain'
    },
    {
      id: 'generate',
      title: 'Generate Classes',
      description: 'AI generates relevant classes',
      completed: ['review', 'complete'].includes(state.currentStep),
      current: state.currentStep === 'generate'
    },
    {
      id: 'review',
      title: 'Review & Edit',
      description: 'Review and customize generated content',
      completed: state.currentStep === 'complete',
      current: state.currentStep === 'review'
    },
    {
      id: 'complete',
      title: 'Complete',
      description: 'Setup complete, choose next steps',
      completed: false,
      current: state.currentStep === 'complete'
    }
  ];

  const handleDomainChange = (domain: string) => {
    setState(prev => ({ ...prev, domain, error: null }));
  };

  const handleClassCountChange = (classCount: number) => {
    setState(prev => ({ ...prev, classCount, error: null }));
  };

  const handleGenerateDataset = async () => {
    setState(prev => ({ ...prev, isLoading: true, error: null, currentStep: 'generate' }));

    try {
      const response = await wizardService.generateDomainDataset(state.domain, {
        classCount: state.classCount,
        includeAttributes: false,
        includeExamples: true,
        examplesPerClass: 1
      });

      if (!response.success) {
        throw new Error(response.error?.message || 'Failed to generate dataset');
      }

      const newDataset = response.data!.dataset;
      
      // Add the new dataset to the store so it's available in other pages
      addDataset(newDataset);
      
      setState(prev => ({
        ...prev,
        dataset: newDataset,
        currentStep: 'review',
        isLoading: false
      }));
    } catch (error) {
      const appError = ErrorHandler.fromUnknown(error);
      setState(prev => ({
        ...prev,
        error: appError.message,
        isLoading: false,
        currentStep: 'domain'
      }));
    }
  };

  const handleDatasetChange = (dataset: Dataset) => {
    setState(prev => ({ ...prev, dataset }));
  };

  const handleSaveAndContinue = async () => {
    if (!state.dataset) return;

    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      // Update the dataset
      const response = await datasetService.saveDataset({
        id: state.dataset.id,
        name: state.dataset.name,
        description: state.dataset.description,
        classes: state.dataset.classes
      });

      if (!response.success) {
        throw new Error(response.error?.message || 'Failed to save dataset');
      }

      setState(prev => ({
        ...prev,
        currentStep: 'complete',
        isLoading: false
      }));
    } catch (error) {
      const appError = ErrorHandler.fromUnknown(error);
      setState(prev => ({
        ...prev,
        error: appError.message,
        isLoading: false
      }));
    }
  };

  const handleBackToReview = () => {
    setState(prev => ({ ...prev, currentStep: 'review' }));
  };

  const handleGoToDatasets = () => {
    navigate('/datasets');
  };

  const handleGoToAttributes = () => {
    if (state.dataset) {
      navigate(`/attributes/${state.dataset.id}`);
    } else {
      navigate('/attributes');
    }
  };

  const handleGoToClassify = () => {
    if (state.dataset) {
      navigate(`/classify/${state.dataset.id}`);
    } else {
      navigate('/classify');
    }
  };

  const handleStartOver = () => {
    setState({
      currentStep: 'domain',
      domain: '',
      classCount: 5,
      dataset: null,
      isLoading: false,
      error: null
    });
  };

  const handleStepClick = (stepId: string) => {
    // Allow navigation to previous steps only
    const stepOrder = ['domain', 'generate', 'review', 'complete'];
    const currentIndex = stepOrder.indexOf(state.currentStep);
    const targetIndex = stepOrder.indexOf(stepId as WizardStep);
    
    if (targetIndex < currentIndex) {
      setState(prev => ({ ...prev, currentStep: stepId as WizardStep }));
    }
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Domain Wizard</h1>
        <p className="text-gray-600 mt-1">
          Quick setup wizard for creating classification scenarios
        </p>
      </div>

      <WizardStepper
        steps={steps}
        currentStep={state.currentStep}
        onStepClick={handleStepClick}
      />

      {state.error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="text-red-800 font-medium">Error</div>
          <div className="text-red-700 text-sm">{state.error}</div>
        </div>
      )}

      {state.currentStep === 'domain' && (
        <DomainInput
          domain={state.domain}
          classCount={state.classCount}
          onDomainChange={handleDomainChange}
          onClassCountChange={handleClassCountChange}
          onNext={handleGenerateDataset}
          isLoading={state.isLoading}
        />
      )}

      {state.currentStep === 'generate' && (
        <div className="flex items-center justify-center py-12">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">Generating Your Dataset</h3>
            <p className="text-gray-600">
              Creating {state.classCount} classes for "{state.domain}" domain...
            </p>
          </div>
        </div>
      )}

      {state.currentStep === 'review' && state.dataset && (
        <GeneratedContent
          dataset={state.dataset}
          onDatasetChange={handleDatasetChange}
          onNext={handleSaveAndContinue}
          onBack={handleBackToReview}
          isLoading={state.isLoading}
        />
      )}

      {state.currentStep === 'complete' && state.dataset && (
        <WizardComplete
          dataset={state.dataset}
          onGoToDatasets={handleGoToDatasets}
          onGoToAttributes={handleGoToAttributes}
          onGoToClassify={handleGoToClassify}
          onStartOver={handleStartOver}
        />
      )}
    </div>
  );
};