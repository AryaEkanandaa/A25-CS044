// src/hooks/useSocket.js - FIXED VERSION

import { useEffect, useRef } from 'react';
import { io } from 'socket.io-client';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5000';

// ✅ Singleton socket instance
let socketInstance = null;

function getSocketInstance() {
  if (!socketInstance) {
    socketInstance = io(API_BASE, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: 5,
      timeout: 10000,
    });

    // Debug logs
    socketInstance.on('connect', () => {
      console.log('✅ Socket connected:', socketInstance.id);
    });

    socketInstance.on('disconnect', (reason) => {
      console.log('🔌 Socket disconnected:', reason);
    });

    socketInstance.on('connect_error', (error) => {
      console.error('❌ Socket connection error:', error.message);
    });
  }
  
  return socketInstance;
}

export default function useSocket(eventHandlers) {
  const handlersRef = useRef(eventHandlers);

  // ✅ Update handlers ref when they change
  useEffect(() => {
    handlersRef.current = eventHandlers;
  }, [eventHandlers]);

  useEffect(() => {
    const socket = getSocketInstance();

    // ✅ Wrapper functions to use latest handlers
    const wrappedHandlers = {};
    
    Object.keys(handlersRef.current).forEach((eventName) => {
      wrappedHandlers[eventName] = (data) => {
        // Use the latest handler from ref
        if (handlersRef.current[eventName]) {
          try {
            handlersRef.current[eventName](data);
          } catch (error) {
            console.error(`Error in ${eventName} handler:`, error);
          }
        }
      };
      
      // ✅ Register event listener
      socket.on(eventName, wrappedHandlers[eventName]);
    });

    // ✅ CLEANUP: Remove event listeners when component unmounts
    return () => {
      Object.keys(wrappedHandlers).forEach((eventName) => {
        socket.off(eventName, wrappedHandlers[eventName]);
      });
      console.log('Cleanup: Socket event listeners removed');
    };
  }, []); // Empty deps - only run once

  return null;
}

// ✅ Cleanup function untuk logout atau app unmount
export function cleanupSocket() {
  if (socketInstance) {
    console.log('Cleanup: Disconnecting socket');
    socketInstance.disconnect();
    socketInstance = null;
  }
}
