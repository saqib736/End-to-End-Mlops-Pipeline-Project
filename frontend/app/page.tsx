'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { UploadCloud, Loader2 } from 'lucide-react'

export default function ImageClassifier() {
  const [image, setImage] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [classification, setClassification] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    setImage(file)
    setPreview(URL.createObjectURL(file))
    setClassification(null)
    setError(null)
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {'image/*': []},
    multiple: false
  })

  const handleSubmit = async () => {
    if (!image) return

    setIsLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('image', image)

    try {
      const response = await fetch('/api/classify', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Classification failed')
      }

      const data = await response.json()
      setClassification(data.classification)
    } catch (err) {
      setError('An error occurred while classifying the image. Please try again.')
      console.error(err)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-400 via-pink-500 to-red-500 flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md"
      >
        <Card className="bg-white/90 backdrop-blur-sm shadow-xl border-none">
          <CardHeader className="space-y-1 pb-2">
            <CardTitle className="text-3xl font-bold text-center bg-clip-text text-transparent bg-gradient-to-r from-purple-500 to-pink-500">
              AI Image Classifier
            </CardTitle>
            <p className="text-center text-gray-500">Upload an image and let AI do its magic!</p>
          </CardHeader>
          <CardContent className="space-y-4">
            <div 
              {...getRootProps()} 
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-300 ${
                isDragActive ? 'border-purple-500 bg-purple-100' : 'border-gray-300 hover:border-pink-500 hover:bg-pink-50'
              }`}
            >
              <input {...getInputProps()} />
              <AnimatePresence mode="wait">
                {preview ? (
                  <motion.img 
                    key="preview"
                    src={preview} 
                    alt="Preview" 
                    className="mx-auto max-h-48 rounded-lg shadow-md" 
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    transition={{ duration: 0.3 }}
                  />
                ) : (
                  <motion.div 
                    key="upload"
                    className="text-gray-500"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <UploadCloud className="mx-auto h-16 w-16 mb-4 text-pink-500" />
                    <p className="font-medium">Drag and drop an image here, or click to select</p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
            <AnimatePresence>
              {classification && (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="p-4 bg-green-100 rounded-lg shadow-inner"
                >
                  <p className="text-green-800 font-medium">{classification}</p>
                </motion.div>
              )}
              {error && (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="p-4 bg-red-100 rounded-lg shadow-inner"
                >
                  <p className="text-red-800 font-medium">{error}</p>
                </motion.div>
              )}
            </AnimatePresence>
          </CardContent>
          <CardFooter>
            <Button 
              onClick={handleSubmit} 
              disabled={!image || isLoading} 
              className="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-semibold py-2 px-4 rounded-md transition-all duration-300 transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-opacity-50 border-none"
            >
              {isLoading ? (
                <span className="flex items-center justify-center">
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Classifying...
                </span>
              ) : (
                'Classify Image'
              )}
            </Button>
          </CardFooter>
        </Card>
      </motion.div>
    </div>
  )
}