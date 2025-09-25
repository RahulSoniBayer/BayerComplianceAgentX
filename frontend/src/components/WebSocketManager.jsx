import React, { createContext, useContext, useCallback, useEffect, useRef } from 'react'

const WebSocketContext = createContext()

export const useWebSocket = () => {
  const context = useContext(WebSocketContext)
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider')
  }
  return context
}

export const WebSocketProvider = ({ children }) => {
  const connections = useRef(new Map())
  const messageHandlers = useRef(new Map())

  const connect = useCallback((taskId) => {
    if (connections.current.has(taskId)) {
      return connections.current.get(taskId)
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const wsUrl = `${protocol}//${host}/api/ws/progress/${taskId}`

    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      console.log(`WebSocket connected for task ${taskId}`)
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        
        // Call registered handlers for this task
        const handlers = messageHandlers.current.get(taskId)
        if (handlers) {
          handlers.forEach(handler => {
            try {
              handler(data)
            } catch (error) {
              console.error('Error in WebSocket message handler:', error)
            }
          })
        }
        
        // Handle ping/pong
        if (data.type === 'ping') {
          ws.send(JSON.stringify({ type: 'pong', timestamp: Date.now() }))
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error)
      }
    }

    ws.onclose = (event) => {
      console.log(`WebSocket disconnected for task ${taskId}:`, event.code, event.reason)
      connections.current.delete(taskId)
      messageHandlers.current.delete(taskId)
    }

    ws.onerror = (error) => {
      console.error(`WebSocket error for task ${taskId}:`, error)
    }

    connections.current.set(taskId, ws)
    return ws
  }, [])

  const disconnect = useCallback((taskId) => {
    const ws = connections.current.get(taskId)
    if (ws) {
      ws.close()
      connections.current.delete(taskId)
      messageHandlers.current.delete(taskId)
    }
  }, [])

  const subscribe = useCallback((taskId, handler) => {
    if (!messageHandlers.current.has(taskId)) {
      messageHandlers.current.set(taskId, new Set())
    }
    
    messageHandlers.current.get(taskId).add(handler)
    
    // Connect if not already connected
    if (!connections.current.has(taskId)) {
      connect(taskId)
    }
    
    // Return unsubscribe function
    return () => {
      const handlers = messageHandlers.current.get(taskId)
      if (handlers) {
        handlers.delete(handler)
        if (handlers.size === 0) {
          messageHandlers.current.delete(taskId)
          disconnect(taskId)
        }
      }
    }
  }, [connect, disconnect])

  const sendMessage = useCallback((taskId, message) => {
    const ws = connections.current.get(taskId)
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message))
    } else {
      console.warn(`WebSocket not connected for task ${taskId}`)
    }
  }, [])

  const isConnected = useCallback((taskId) => {
    const ws = connections.current.get(taskId)
    return ws && ws.readyState === WebSocket.OPEN
  }, [])

  // Cleanup all connections on unmount
  useEffect(() => {
    return () => {
      connections.current.forEach((ws, taskId) => {
        ws.close()
      })
      connections.current.clear()
      messageHandlers.current.clear()
    }
  }, [])

  const value = {
    connect,
    disconnect,
    subscribe,
    sendMessage,
    isConnected
  }

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  )
}

// Hook for managing WebSocket connections with automatic cleanup
export const useWebSocketConnection = (taskId, onMessage) => {
  const { subscribe, isConnected } = useWebSocket()

  useEffect(() => {
    if (!taskId || !onMessage) return

    const unsubscribe = subscribe(taskId, onMessage)
    return unsubscribe
  }, [taskId, onMessage, subscribe])

  return {
    isConnected: isConnected(taskId)
  }
}

// Hook for sending WebSocket messages
export const useWebSocketSend = () => {
  const { sendMessage } = useWebSocket()
  return sendMessage
}

// Message types and utilities
export const WebSocketMessageTypes = {
  PROGRESS_UPDATE: 'progress_update',
  FILE_COMPLETED: 'file_completed',
  TASK_COMPLETED: 'task_completed',
  ERROR: 'error',
  HEARTBEAT: 'heartbeat',
  PING: 'ping',
  PONG: 'pong'
}

export const createProgressMessage = (taskId, status, message, progress = 0, extra = {}) => ({
  type: WebSocketMessageTypes.PROGRESS_UPDATE,
  task_id: taskId,
  status,
  message,
  progress,
  timestamp: Date.now(),
  ...extra
})

export const createFileCompletedMessage = (taskId, fileIndex, filename, completedFiles, totalFiles, downloadUrl) => ({
  type: WebSocketMessageTypes.FILE_COMPLETED,
  task_id: taskId,
  file_index: fileIndex,
  filename,
  completed_files: completedFiles,
  total_files: totalFiles,
  download_url: downloadUrl,
  timestamp: Date.now()
})

export const createTaskCompletedMessage = (taskId, success, message = '', extra = {}) => ({
  type: WebSocketMessageTypes.TASK_COMPLETED,
  task_id: taskId,
  success,
  message,
  timestamp: Date.now(),
  ...extra
})

export const createErrorMessage = (taskId, error, details = {}) => ({
  type: WebSocketMessageTypes.ERROR,
  task_id: taskId,
  error,
  details,
  timestamp: Date.now()
})
