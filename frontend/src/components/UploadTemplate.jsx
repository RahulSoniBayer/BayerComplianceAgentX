import React, { useState, useCallback, useRef } from 'react'
import { Upload, File, X, CheckCircle, AlertCircle, Image } from 'lucide-react'
import { Button } from './ui/Button'
import { Card, CardContent } from './ui/Card'
import { formatFileSize, validateFileType, validateFileSize } from '../lib/utils'

const UploadTemplate = ({ onUpload, isLoading = false }) => {
  const [dragActive, setDragActive] = useState(false)
  const [selectedFiles, setSelectedFiles] = useState([])
  const [userContext, setUserContext] = useState('')
  const [processFlowImage, setProcessFlowImage] = useState(null)
  const [processFlowPreview, setProcessFlowPreview] = useState(null)
  const fileInputRef = useRef(null)
  const imageInputRef = useRef(null)

  const allowedTypes = ['docx', 'doc']
  const maxSizeMB = 50

  const validateFile = useCallback((file) => {
    const errors = []
    
    if (!validateFileType(file, allowedTypes)) {
      errors.push('File type not allowed. Only Word documents (.docx, .doc) are supported.')
    }
    
    if (!validateFileSize(file, maxSizeMB)) {
      errors.push(`File size exceeds ${maxSizeMB}MB limit.`)
    }
    
    return {
      valid: errors.length === 0,
      errors
    }
  }, [])

  const handleDrag = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    const files = Array.from(e.dataTransfer.files)
    const docxFiles = files.filter(file => allowedTypes.includes(file.name.split('.').pop().toLowerCase()))
    
    if (docxFiles.length > 0) {
      handleFiles(docxFiles)
    }
  }, [])

  const handleFiles = useCallback((files) => {
    const newFiles = files.map(file => {
      const validation = validateFile(file)
      return {
        id: Math.random().toString(36).substr(2, 9),
        file,
        name: file.name,
        size: file.size,
        valid: validation.valid,
        errors: validation.errors,
        status: validation.valid ? 'ready' : 'error'
      }
    })
    setSelectedFiles(prev => [...prev, ...newFiles])
  }, [validateFile])

  const handleFileInput = useCallback((e) => {
    const files = Array.from(e.target.files)
    handleFiles(files)
    e.target.value = '' // Reset input
  }, [handleFiles])

  const handleChooseTemplates = useCallback(() => {
    fileInputRef.current?.click()
  }, [])

  const removeFile = useCallback((fileId) => {
    setSelectedFiles(prev => prev.filter(f => f.id !== fileId))
  }, [])

  const handleImageUpload = useCallback((e) => {
    const file = e.target.files[0]
    if (!file) return

    const allowedImageTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp']
    if (!allowedImageTypes.includes(file.type)) {
      alert('Please select a valid image file (JPEG, PNG, GIF, BMP)')
      return
    }
    if (file.size > 10 * 1024 * 1024) {
      alert('Image file size must be less than 10MB')
      return
    }

    const reader = new FileReader()
    reader.onload = (event) => {
      const base64 = event.target.result.split(',')[1]
      setProcessFlowImage(base64)
      setProcessFlowPreview(event.target.result)
    }
    reader.readAsDataURL(file)
  }, [])

  const removeImage = useCallback(() => {
    setProcessFlowImage(null)
    setProcessFlowPreview(null)
    if (imageInputRef.current) imageInputRef.current.value = ''
  }, [])

  const handleUpload = useCallback(async () => {
    const validFiles = selectedFiles.filter(f => f.valid)
    if (validFiles.length === 0) {
      alert('Please select at least one valid template file')
      return
    }

    setSelectedFiles(prev => prev.map(f => f.valid ? { ...f, status: 'uploading' } : f))

    try {
      await onUpload({
        files: validFiles.map(f => f.file),
        userContext: userContext.trim() || null,
        processFlowImage: processFlowImage || null
      })

      setSelectedFiles(prev => prev.map(f => f.status === 'uploading' ? { ...f, status: 'uploaded' } : f))
    } catch (error) {
      setSelectedFiles(prev => prev.map(f => f.status === 'uploading' ? { ...f, status: 'error', errors: [error.message] } : f))
    }
  }, [selectedFiles, userContext, processFlowImage, onUpload])

  const clearAll = useCallback(() => {
    setSelectedFiles([])
    setUserContext('')
    removeImage()
  }, [removeImage])

  const validFiles = selectedFiles.filter(f => f.valid)
  const hasValidFiles = validFiles.length > 0

  return (
    <Card className="w-full">
      <CardContent className="p-6">
        <div className="space-y-6">
          {/* Upload Area */}
          <div
            className={`upload-area ${dragActive ? 'dragover' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <div className="flex flex-col items-center space-y-4">
              <div className="p-4 bg-green-100 rounded-full">
                <Upload className="h-8 w-8 text-green-600" />
              </div>

              <div className="text-center">
                <h3 className="text-lg font-semibold text-gray-900">
                  Upload Template Files
                </h3>
                <p className="text-sm text-gray-500 mt-1">
                  Drag and drop your Word template files here, or click to browse
                </p>
                <p className="text-xs text-gray-400 mt-2">
                  Maximum file size: {maxSizeMB}MB per file
                </p>
              </div>

              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".docx,.doc"
                onChange={handleFileInput}
                style={{ display: 'none' }}
              />
              <Button variant="outline" onClick={handleChooseTemplates}>
                Choose Templates
              </Button>
            </div>
          </div>

          {/* File List */}
          {selectedFiles.length > 0 && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-medium text-gray-900">
                  Selected Templates ({selectedFiles.length})
                </h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={clearAll}
                  className="text-gray-500 hover:text-gray-700"
                >
                  Clear All
                </Button>
              </div>

              <div className="space-y-2 max-h-64 overflow-y-auto">
                {selectedFiles.map(file => (
                  <div key={file.id} className="file-list-item flex items-center justify-between">
                    <div className="flex items-center space-x-3 flex-1">
                      <File className="h-5 w-5 text-gray-400" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 truncate">{file.name}</p>
                        <p className="text-xs text-gray-500">{formatFileSize(file.size)}</p>
                        {file.errors && file.errors.length > 0 && (
                          <div className="mt-1">
                            {file.errors.map((error, index) => (
                              <p key={index} className="text-xs text-red-600">{error}</p>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="flex items-center space-x-2">
                      {file.status === 'ready' && <CheckCircle className="h-5 w-5 text-green-500" />}
                      {file.status === 'uploading' && <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>}
                      {file.status === 'uploaded' && <CheckCircle className="h-5 w-5 text-green-500" />}
                      {file.status === 'error' && <AlertCircle className="h-5 w-5 text-red-500" />}

                      <Button variant="ghost" size="sm" onClick={() => removeFile(file.id)} className="h-8 w-8 p-0 text-gray-400 hover:text-red-600">
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* User Context */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700">Additional Context (Optional)</label>
            <div className="relative">
              <textarea
                placeholder="Provide any additional context or instructions for filling the templates..."
                value={userContext}
                onChange={e => setUserContext(e.target.value)}
                className="min-h-[100px] resize-none w-full border rounded-md p-2"
                maxLength={5000}
              />
              <div className="absolute bottom-2 right-2 text-xs text-gray-400">
                {userContext.length}/5000
              </div>
            </div>
          </div>

          {/* Process Flow Image */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700">Process Flow Image (Optional)</label>
            <div className="space-y-3">
              {!processFlowPreview ? (
                <div
                  className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center cursor-pointer hover:border-gray-400 transition-colors"
                  onClick={() => imageInputRef.current?.click()}
                >
                  <Image className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                  <p className="text-sm text-gray-600">Click to upload process flow image</p>
                  <p className="text-xs text-gray-400">JPEG, PNG, GIF, BMP (max 10MB)</p>
                </div>
              ) : (
                <div className="relative">
                  <img
                    src={processFlowPreview}
                    alt="Process flow"
                    className="w-full h-48 object-contain border border-gray-200 rounded-lg"
                  />
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={removeImage}
                    className="absolute top-2 right-2 h-8 w-8 p-0 bg-white/80 hover:bg-white"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              )}
              <input
                ref={imageInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                style={{ display: 'none' }}
              />
            </div>
          </div>

          {/* Upload Button */}
          {hasValidFiles && (
            <div className="flex items-center justify-between pt-4 border-t">
              <div className="text-sm text-gray-600">{validFiles.length} template(s) ready to process</div>
              <Button onClick={handleUpload} disabled={isLoading || !hasValidFiles} className="min-w-32">
                {isLoading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Processing...
                  </>
                ) : (
                  `Generate ${validFiles.length} Document${validFiles.length > 1 ? 's' : ''}`
                )}
              </Button>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

export default UploadTemplate
