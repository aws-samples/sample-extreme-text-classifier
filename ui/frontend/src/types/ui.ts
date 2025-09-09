/**
 * UI-related TypeScript interfaces and types
 */

import { AppError } from './errors';

// Loading states
export interface LoadingState {
  isLoading: boolean;
  operation?: string;
  progress?: number;
  message?: string;
}

// Toast notification types
export type ToastType = 'success' | 'error' | 'warning' | 'info';

export interface Toast {
  id: string;
  type: ToastType;
  title: string;
  message?: string;
  duration?: number;
  actions?: Array<{
    label: string;
    action: () => void;
  }>;
}

// Modal types
export interface ModalState {
  isOpen: boolean;
  type?: 'confirm' | 'info' | 'error' | 'custom';
  title?: string;
  content?: string;
  onConfirm?: () => void;
  onCancel?: () => void;
  confirmText?: string;
  cancelText?: string;
}

// Navigation types
export type NavigationSection = 'dashboard' | 'datasets' | 'attributes' | 'classify' | 'wizard';

export interface NavigationItem {
  id: NavigationSection;
  label: string;
  path: string;
  icon?: string;
  badge?: string | number;
}

// Theme types
export type Theme = 'light' | 'dark' | 'system';

// UI preferences
export interface UIPreferences {
  theme: Theme;
  sidebarCollapsed: boolean;
  showTooltips: boolean;
  animationsEnabled: boolean;
  compactMode: boolean;
}

// Form states
export interface FormState<T = any> {
  data: T;
  errors: Record<string, string>;
  touched: Record<string, boolean>;
  isSubmitting: boolean;
  isValid: boolean;
}

// Table/List states
export interface TableState {
  sortBy?: string;
  sortOrder: 'asc' | 'desc';
  currentPage: number;
  pageSize: number;
  totalItems: number;
  filters: Record<string, any>;
  searchQuery: string;
}

// File upload states
export interface FileUploadState {
  file?: File;
  progress: number;
  status: 'idle' | 'uploading' | 'success' | 'error';
  error?: AppError;
}

// Wizard states
export interface WizardState {
  currentStep: number;
  totalSteps: number;
  canGoNext: boolean;
  canGoPrevious: boolean;
  isComplete: boolean;
  stepData: Record<number, any>;
}

