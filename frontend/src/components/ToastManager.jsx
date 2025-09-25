import React, { createContext, useContext, useCallback, useState } from 'react'
import { Toast, ToastProvider, ToastViewport, ToastTitle, ToastDescription, ToastClose } from './ui/Toast'

const ToastContext = createContext()

export const useToast = () => {
  const context = useContext(ToastContext)
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider')
  }
  return context
}

export const ToastManager = ({ children }) => {
  const [toasts, setToasts] = useState([])

  const addToast = useCallback((toast) => {
    const id = Math.random().toString(36).substr(2, 9)
    const newToast = {
      id,
      ...toast,
      open: true
    }
    
    setToasts(prev => [...prev, newToast])
    
    // Auto-remove after duration (default 5 seconds)
    const duration = toast.duration || 5000
    if (duration > 0) {
      setTimeout(() => {
        removeToast(id)
      }, duration)
    }
    
    return id
  }, [])

  const removeToast = useCallback((id) => {
    setToasts(prev => prev.filter(toast => toast.id !== id))
  }, [])

  const showSuccess = useCallback((title, description, duration = 5000) => {
    return addToast({
      title,
      description,
      variant: 'default',
      className: 'toast-success',
      duration
    })
  }, [addToast])

  const showError = useCallback((title, description, duration = 8000) => {
    return addToast({
      title,
      description,
      variant: 'destructive',
      className: 'toast-error',
      duration
    })
  }, [addToast])

  const showInfo = useCallback((title, description, duration = 5000) => {
    return addToast({
      title,
      description,
      variant: 'default',
      className: 'toast-info',
      duration
    })
  }, [addToast])

  const showWarning = useCallback((title, description, duration = 6000) => {
    return addToast({
      title,
      description,
      variant: 'default',
      className: 'toast-warning',
      duration
    })
  }, [addToast])

  const value = {
    addToast,
    removeToast,
    showSuccess,
    showError,
    showInfo,
    showWarning
  }

  return (
    <ToastContext.Provider value={value}>
      <ToastProvider>
        {children}
        <ToastViewport />
        {toasts.map((toast) => (
          <Toast
            key={toast.id}
            open={toast.open}
            onOpenChange={(open) => {
              if (!open) {
                removeToast(toast.id)
              }
            }}
            variant={toast.variant}
            className={toast.className}
          >
            <div className="grid gap-1">
              {toast.title && (
                <ToastTitle>{toast.title}</ToastTitle>
              )}
              {toast.description && (
                <ToastDescription>{toast.description}</ToastDescription>
              )}
            </div>
            <ToastClose />
          </Toast>
        ))}
      </ToastProvider>
    </ToastContext.Provider>
  )
}

// Convenience hooks for specific toast types
export const useSuccessToast = () => {
  const { showSuccess } = useToast()
  return showSuccess
}

export const useErrorToast = () => {
  const { showError } = useToast()
  return showError
}

export const useInfoToast = () => {
  const { showInfo } = useToast()
  return showInfo
}

export const useWarningToast = () => {
  const { showWarning } = useToast()
  return showWarning
}

// Specialized toast for file operations
export const useFileToast = () => {
  const { showSuccess, showError, showInfo } = useToast()

  const showFileUploaded = useCallback((filename) => {
    showSuccess(
      'File Uploaded',
      `${filename} has been uploaded successfully.`
    )
  }, [showSuccess])

  const showFileProcessed = useCallback((filename) => {
    showSuccess(
      'File Generated',
      `${filename} has been generated successfully.`
    )
  }, [showSuccess])

  const showFileError = useCallback((filename, error) => {
    showError(
      'File Error',
      `Error processing ${filename}: ${error}`
    )
  }, [showError])

  const showUploadProgress = useCallback((filename, progress) => {
    showInfo(
      'Upload Progress',
      `Uploading ${filename}: ${progress}% complete`
    )
  }, [showInfo])

  const showProcessingProgress = useCallback((filename, progress) => {
    showInfo(
      'Processing Progress',
      `Processing ${filename}: ${progress}% complete`
    )
  }, [showInfo])

  return {
    showFileUploaded,
    showFileProcessed,
    showFileError,
    showUploadProgress,
    showProcessingProgress
  }
}

// Specialized toast for batch operations
export const useBatchToast = () => {
  const { showSuccess, showError, showInfo } = useToast()

  const showBatchStarted = useCallback((count) => {
    showInfo(
      'Batch Processing Started',
      `Processing ${count} file${count > 1 ? 's' : ''}...`
    )
  }, [showInfo])

  const showBatchCompleted = useCallback((completed, total) => {
    showSuccess(
      'Batch Processing Completed',
      `Successfully processed ${completed} of ${total} files.`
    )
  }, [showSuccess])

  const showBatchError = useCallback((error) => {
    showError(
      'Batch Processing Error',
      `Error in batch processing: ${error}`
    )
  }, [showError])

  const showBatchProgress = useCallback((completed, total) => {
    const progress = Math.round((completed / total) * 100)
    showInfo(
      'Batch Progress',
      `${completed} of ${total} files completed (${progress}%)`
    )
  }, [showInfo])

  return {
    showBatchStarted,
    showBatchCompleted,
    showBatchError,
    showBatchProgress
  }
}
