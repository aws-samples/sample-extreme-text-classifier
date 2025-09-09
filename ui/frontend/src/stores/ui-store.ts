/**
 * UI Store - Manages application state (loading, errors, navigation) using Zustand
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { 
  LoadingState, 
  Toast, 
  ModalState, 
  NavigationSection, 
  NavigationItem,
  Theme,
  UIPreferences,
  FormState,
  TableState,
  FileUploadState,
  WizardState
} from '../types/ui';
import { AppError } from '../types/errors';

interface UIState {
  // Loading state
  loading: LoadingState;
  
  // Toast notifications
  toasts: Toast[];
  
  // Modal state
  modal: ModalState;
  
  // Navigation
  currentSection: NavigationSection;
  navigationItems: NavigationItem[];
  
  // UI preferences
  preferences: UIPreferences;
  
  // Form states
  forms: Record<string, FormState>;
  
  // Table states
  tables: Record<string, TableState>;
  
  // File upload states
  uploads: Record<string, FileUploadState>;
  
  // Wizard state
  wizard: WizardState;
  

  
  // Global error state
  globalError: AppError | null;

  // Loading actions
  setLoading: (loading: Partial<LoadingState>) => void;
  clearLoading: () => void;
  
  // Toast actions
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
  clearToasts: () => void;
  
  // Modal actions
  openModal: (modal: Partial<ModalState>) => void;
  closeModal: () => void;
  
  // Navigation actions
  setCurrentSection: (section: NavigationSection) => void;
  updateNavigationBadge: (sectionId: NavigationSection, badge?: string | number) => void;
  
  // Preferences actions
  updatePreferences: (updates: Partial<UIPreferences>) => void;
  setTheme: (theme: Theme) => void;
  toggleSidebar: () => void;
  
  // Form actions
  setFormState: (formId: string, state: FormState) => void;
  updateFormData: (formId: string, data: any) => void;
  setFormErrors: (formId: string, errors: Record<string, string>) => void;
  clearForm: (formId: string) => void;
  
  // Table actions
  setTableState: (tableId: string, state: TableState) => void;
  updateTableSort: (tableId: string, sortBy: string, sortOrder: 'asc' | 'desc') => void;
  updateTableFilters: (tableId: string, filters: Record<string, any>) => void;
  setTablePage: (tableId: string, page: number) => void;
  
  // File upload actions
  setUploadState: (uploadId: string, state: FileUploadState) => void;
  updateUploadProgress: (uploadId: string, progress: number) => void;
  clearUpload: (uploadId: string) => void;
  
  // Wizard actions
  setWizardStep: (step: number) => void;
  setWizardData: (step: number, data: any) => void;
  resetWizard: () => void;
  

  
  // Error actions
  setGlobalError: (error: AppError | null) => void;
  clearGlobalError: () => void;
  
  // Utility actions
  reset: () => void;
}

// Default states
const defaultPreferences: UIPreferences = {
  theme: 'system',
  sidebarCollapsed: false,
  showTooltips: true,
  animationsEnabled: true,
  compactMode: false
};

const defaultNavigationItems: NavigationItem[] = [
  { id: 'dashboard', label: 'Dashboard', path: '/' },
  { id: 'datasets', label: 'Datasets', path: '/datasets' },
  { id: 'attributes', label: 'Attributes', path: '/attributes' },
  { id: 'classify', label: 'Classify', path: '/classify' },
  { id: 'wizard', label: 'Wizard', path: '/wizard' }
];

const defaultWizardState: WizardState = {
  currentStep: 0,
  totalSteps: 0,
  canGoNext: false,
  canGoPrevious: false,
  isComplete: false,
  stepData: {}
};



export const useUIStore = create<UIState>()(
  persist(
    (set, get) => ({
      // Initial state
      loading: { isLoading: false },
      toasts: [],
      modal: { isOpen: false },
      currentSection: 'dashboard',
      navigationItems: defaultNavigationItems,
      preferences: defaultPreferences,
      forms: {},
      tables: {},
      uploads: {},
      wizard: defaultWizardState,
      globalError: null,

      // Loading actions
      setLoading: (loading) => set((state) => ({
        loading: { ...state.loading, ...loading }
      })),
      clearLoading: () => set({ loading: { isLoading: false } }),

      // Toast actions
      addToast: (toast) => {
        const id = `toast_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
        const newToast: Toast = { ...toast, id };
        set((state) => ({ toasts: [...state.toasts, newToast] }));
        
        // Auto-remove toast after duration
        if (toast.duration !== 0) {
          setTimeout(() => {
            get().removeToast(id);
          }, toast.duration || 5000);
        }
      },
      removeToast: (id) => set((state) => ({
        toasts: state.toasts.filter(t => t.id !== id)
      })),
      clearToasts: () => set({ toasts: [] }),

      // Modal actions
      openModal: (modal) => set((state) => ({
        modal: { ...state.modal, ...modal, isOpen: true }
      })),
      closeModal: () => set({ modal: { isOpen: false } }),

      // Navigation actions
      setCurrentSection: (section) => set({ currentSection: section }),
      updateNavigationBadge: (sectionId, badge) => set((state) => ({
        navigationItems: state.navigationItems.map(item =>
          item.id === sectionId ? { ...item, badge } : item
        )
      })),

      // Preferences actions
      updatePreferences: (updates) => set((state) => ({
        preferences: { ...state.preferences, ...updates }
      })),
      setTheme: (theme) => set((state) => ({
        preferences: { ...state.preferences, theme }
      })),
      toggleSidebar: () => set((state) => ({
        preferences: { 
          ...state.preferences, 
          sidebarCollapsed: !state.preferences.sidebarCollapsed 
        }
      })),

      // Form actions
      setFormState: (formId, state) => set((prevState) => ({
        forms: { ...prevState.forms, [formId]: state }
      })),
      updateFormData: (formId, data) => set((state) => ({
        forms: {
          ...state.forms,
          [formId]: {
            ...state.forms[formId],
            data: { ...state.forms[formId]?.data, ...data }
          }
        }
      })),
      setFormErrors: (formId, errors) => set((state) => ({
        forms: {
          ...state.forms,
          [formId]: {
            ...state.forms[formId],
            errors
          }
        }
      })),
      clearForm: (formId) => set((state) => {
        const { [formId]: removed, ...rest } = state.forms;
        return { forms: rest };
      }),

      // Table actions
      setTableState: (tableId, state) => set((prevState) => ({
        tables: { ...prevState.tables, [tableId]: state }
      })),
      updateTableSort: (tableId, sortBy, sortOrder) => set((state) => ({
        tables: {
          ...state.tables,
          [tableId]: {
            ...state.tables[tableId],
            sortBy,
            sortOrder
          }
        }
      })),
      updateTableFilters: (tableId, filters) => set((state) => ({
        tables: {
          ...state.tables,
          [tableId]: {
            ...state.tables[tableId],
            filters: { ...state.tables[tableId]?.filters, ...filters }
          }
        }
      })),
      setTablePage: (tableId, page) => set((state) => ({
        tables: {
          ...state.tables,
          [tableId]: {
            ...state.tables[tableId],
            currentPage: page
          }
        }
      })),

      // File upload actions
      setUploadState: (uploadId, state) => set((prevState) => ({
        uploads: { ...prevState.uploads, [uploadId]: state }
      })),
      updateUploadProgress: (uploadId, progress) => set((state) => ({
        uploads: {
          ...state.uploads,
          [uploadId]: {
            ...state.uploads[uploadId],
            progress
          }
        }
      })),
      clearUpload: (uploadId) => set((state) => {
        const { [uploadId]: removed, ...rest } = state.uploads;
        return { uploads: rest };
      }),

      // Wizard actions
      setWizardStep: (step) => set((state) => ({
        wizard: {
          ...state.wizard,
          currentStep: step,
          canGoPrevious: step > 0,
          canGoNext: step < state.wizard.totalSteps - 1,
          isComplete: step >= state.wizard.totalSteps - 1
        }
      })),
      setWizardData: (step, data) => set((state) => ({
        wizard: {
          ...state.wizard,
          stepData: { ...state.wizard.stepData, [step]: data }
        }
      })),
      resetWizard: () => set({ wizard: defaultWizardState }),



      // Error actions
      setGlobalError: (error) => set({ globalError: error }),
      clearGlobalError: () => set({ globalError: null }),

      // Utility actions
      reset: () => set({
        loading: { isLoading: false },
        toasts: [],
        modal: { isOpen: false },
        currentSection: 'dashboard',
        navigationItems: defaultNavigationItems,
        preferences: defaultPreferences,
        forms: {},
        tables: {},
        uploads: {},
        wizard: defaultWizardState,
        globalError: null
      })
    }),
    {
      name: 'ui-store',
      partialize: (state) => ({
        preferences: state.preferences,
        currentSection: state.currentSection
      })
    }
  )
);