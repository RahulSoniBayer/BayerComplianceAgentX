import React, { useState, useCallback, useEffect } from 'react'
import { Upload, FileText, Download, Settings, RefreshCw } from 'lucide-react'
import { Button } from '../components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card'
import UploadPDF from '../components/UploadPDF'
import UploadTemplate from '../components/UploadTemplate'
import FileProgressList from '../components/FileProgressList'
import { WebSocketProvider, useWebSocketConnection } from '../components/WebSocketManager'
import { ToastManager, useFileToast, useBatchToast } from '../components/ToastManager'
import { formatFileSize } from '../lib/utils'

const API_BASE_URL = process.env.REACT_APP_API_URL || ''

const Home = () => {
  const [activeTab, setActiveTab] = useState('pdf')
  const [isUploadingPDF, setIsUploadingPDF] = useState(false)
  const [isUploadingTemplate, setIsUploadingTemplate] = useState(false)
  const [currentTask, setCurrentTask] = useState(null)
  const [taskFiles, setTaskFiles] = useState([])
  const [pdfFiles, setPdfFiles] = useState([])
  const [isLoadingPdfs, setIsLoadingPdfs] = useState(false)

  const { showFileToast } = useFileToast()
  const { showBatchToast } = useBatchToast()

  // Load PDF files on component mount
  useEffect(() => {
    loadPdfFiles()
  }, [])

  const loadPdfFiles = useCallback(async () => {
    setIsLoadingPdfs(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/pdf/list`)
      const data = await response.json()
      
      if (data.success) {
        setPdfFiles(data.documents || [])
      } else {
        console.error('Failed to load PDF files:', data.message)
      }
    } catch (error) {
      console.error('Error loading PDF files:', error)
    } finally {
      setIsLoadingPdfs(false)
    }
  }, [])

  const handlePdfUpload = useCallback(async (files) => {
    setIsUploadingPDF(true)
    
    try {
      for (const file of files) {
        const formData = new FormData()
        formData.append('file', file)
        
        const response = await fetch(`${API_BASE_URL}/api/pdf/upload`, {
          method: 'POST',
          body: formData
        })
        
        const data = await response.json()
        
        if (data.success) {
          showFileToast.showFileUploaded(file.name)
        } else {
          showFileToast.showFileError(file.name, data.message || 'Upload failed')
        }
      }
      
      // Reload PDF files
      await loadPdfFiles()
    } catch (error) {
      console.error('Error uploading PDFs:', error)
      showFileToast.showFileError('PDF Upload', error.message)
    } finally {
      setIsUploadingPDF(false)
    }
  }, [loadPdfFiles, showFileToast])

  const handleTemplateUpload = useCallback(async (uploadData) => {
    setIsUploadingTemplate(true)
    
    try {
      const formData = new FormData()
      
      // Add files
      uploadData.files.forEach(file => {
        formData.append('files', file)
      })
      
      // Add user context
      if (uploadData.userContext) {
        formData.append('user_context', uploadData.userContext)
      }
      
      // Add process flow image
      if (uploadData.processFlowImage) {
        formData.append('process_flow_image', uploadData.processFlowImage)
      }
      
      const response = await fetch(`${API_BASE_URL}/api/template/upload`, {
        method: 'POST',
        body: formData
      })
      
      const data = await response.json()
      
      if (data.success) {
        setCurrentTask(data.task_id)
        showBatchToast.showBatchStarted(data.total_files)
        
        // Initialize task files
        setTaskFiles(uploadData.files.map((file, index) => ({
          id: index,
          filename: file.name,
          original_filename: file.name,
          status: 'pending',
          progress: 0
        })))
      } else {
        showBatchToast.showBatchError(data.message || 'Upload failed')
      }
    } catch (error) {
      console.error('Error uploading templates:', error)
      showBatchToast.showBatchError(error.message)
    } finally {
      setIsUploadingTemplate(false)
    }
  }, [showBatchToast])

  const handleDownloadSingle = useCallback(async (fileId, filename) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/template/download/${fileId}`)
      
      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = filename || 'generated_document.docx'
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)
        
        showFileToast.showFileProcessed(filename)
      } else {
        throw new Error('Download failed')
      }
    } catch (error) {
      console.error('Error downloading file:', error)
      showFileToast.showFileError(filename, error.message)
    }
  }, [showFileToast])

  const handleDownloadAll = useCallback(async (taskId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/template/download_all/${taskId}`)
      
      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `generated_documents_${taskId}.zip`
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)
        
        showFileToast.showFileProcessed('All Documents')
      } else {
        throw new Error('Download failed')
      }
    } catch (error) {
      console.error('Error downloading all files:', error)
      showFileToast.showFileError('Download All', error.message)
    }
  }, [showFileToast])

  // WebSocket message handler
  const handleWebSocketMessage = useCallback((data) => {
    console.log('WebSocket message:', data)
    
    switch (data.type) {
      case 'progress_update':
        setTaskFiles(prev => prev.map(file => {
          if (data.file_index === file.id) {
            return {
              ...file,
              status: data.status === 'processing_file' ? 'processing' : file.status,
              progress: data.progress || file.progress
            }
          }
          return file
        }))
        break
        
      case 'file_completed':
        setTaskFiles(prev => prev.map(file => {
          if (data.file_index === file.id) {
            return {
              ...file,
              status: 'completed',
              progress: 100,
              generated_filename: data.filename,
              download_url: data.download_url
            }
          }
          return file
        }))
        
        showFileToast.showFileProcessed(data.filename)
        showBatchToast.showBatchProgress(data.completed_files, data.total_files)
        break
        
      case 'task_completed':
        if (data.success) {
          showBatchToast.showBatchCompleted(data.completed_files, data.total_files)
        } else {
          showBatchToast.showBatchError(data.message || 'Task failed')
        }
        break
        
      case 'error':
        console.error('WebSocket error:', data.error)
        showBatchToast.showBatchError(data.error)
        break
    }
  }, [showFileToast, showBatchToast])

  // Set up WebSocket connection for current task
  useWebSocketConnection(currentTask, handleWebSocketMessage)

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Bayer Compliance Agent
          </h1>
          <p className="text-gray-600">
            AI-assisted document automation platform for filling templates with retrieved content
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="mb-8">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8">
              <button
                onClick={() => setActiveTab('pdf')}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'pdf'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <div className="flex items-center space-x-2">
                  <FileText className="h-4 w-4" />
                  <span>Reference PDFs</span>
                </div>
              </button>
              
              <button
                onClick={() => setActiveTab('template')}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'template'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <div className="flex items-center space-x-2">
                  <Upload className="h-4 w-4" />
                  <span>Template Processing</span>
                </div>
              </button>
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        <div className="space-y-8">
          {activeTab === 'pdf' && (
            <div className="space-y-6">
              {/* PDF Upload */}
              <UploadPDF
                onUpload={handlePdfUpload}
                isLoading={isUploadingPDF}
              />
              
              {/* PDF Files List */}
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center space-x-2">
                      <FileText className="h-5 w-5" />
                      <span>Reference Documents</span>
                    </CardTitle>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={loadPdfFiles}
                      disabled={isLoadingPdfs}
                    >
                      <RefreshCw className={`h-4 w-4 ${isLoadingPdfs ? 'animate-spin' : ''}`} />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  {isLoadingPdfs ? (
                    <div className="text-center py-8">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
                      <p className="text-gray-500 mt-2">Loading documents...</p>
                    </div>
                  ) : pdfFiles.length === 0 ? (
                    <div className="text-center py-8 text-gray-500">
                      <FileText className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                      <p>No reference documents uploaded yet</p>
                      <p className="text-sm">Upload PDF documents to use as reference content</p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {pdfFiles.map((pdf) => (
                        <div key={pdf.id} className="flex items-center justify-between p-3 bg-white border border-gray-200 rounded-lg">
                          <div className="flex items-center space-x-3">
                            <FileText className="h-5 w-5 text-red-500" />
                            <div>
                              <p className="text-sm font-medium text-gray-900">
                                {pdf.filename}
                              </p>
                              <p className="text-xs text-gray-500">
                                {formatFileSize(pdf.file_size)} • {pdf.chunk_count || 0} chunks
                              </p>
                            </div>
                          </div>
                          
                          <div className="flex items-center space-x-2">
                            <span className={`status-badge ${
                              pdf.processing_status === 'completed' ? 'status-completed' :
                              pdf.processing_status === 'processing' ? 'status-processing' :
                              pdf.processing_status === 'failed' ? 'status-failed' : 'status-pending'
                            }`}>
                              {pdf.processing_status}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          )}
          
          {activeTab === 'template' && (
            <div className="space-y-6">
              {/* Template Upload */}
              <UploadTemplate
                onUpload={handleTemplateUpload}
                isLoading={isUploadingTemplate}
              />
              
              {/* Progress List */}
              {currentTask && (
                <FileProgressList
                  taskId={currentTask}
                  files={taskFiles}
                  onDownloadAll={handleDownloadAll}
                  onDownloadSingle={handleDownloadSingle}
                />
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

const HomeWithProviders = () => (
  <WebSocketProvider>
    <ToastManager>
      <Home />
    </ToastManager>
  </WebSocketProvider>
)

export default HomeWithProviders
