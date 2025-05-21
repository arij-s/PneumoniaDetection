import React, { useState } from 'react';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', image);

    try {
      // Update this URL if your backend is running on a different port
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        body: formData,
        // Don't set Content-Type header - let the browser set it with boundary
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to analyze image');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'An error occurred during analysis');
      console.error('API Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Pneumonia Detection</h1>
        <p>Upload a chest X-ray to analyze for pneumonia</p>
      </header>

      <main className="main-content">
        <div className="upload-container">
          <form onSubmit={handleSubmit} encType="multipart/form-data">
            <div className="file-upload">
              <label htmlFor="xray-upload" className="upload-label">
                {preview ? (
                  <img src={preview} alt="Preview" className="image-preview" />
                ) : (
                  <div className="upload-placeholder">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                      <polyline points="17 8 12 3 7 8"></polyline>
                      <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                    <span>Click to upload X-ray</span>
                  </div>
                )}
                <input
                  id="xray-upload"
                  type="file"
                  accept=".png,.jpg,.jpeg"
                  onChange={handleImageChange}
                  className="hidden-input"
                />
              </label>
            </div>

            <button type="submit" className="analyze-button" disabled={loading || !image}>
              {loading ? 'Analyzing...' : 'Analyze X-ray'}
            </button>
          </form>

          {error && <div className="error-message">{error}</div>}
        </div>

        {result && (
          <div className={`result-container ${result.prediction ? 'pneumonia' : 'normal'}`}>
            <h2>Analysis Result</h2>
            <div className="result-content">
              <div className="result-image">
                <img src={`http://localhost:5000${result.image_path}`} alt="Analyzed X-ray" />
              </div>
              <div className="result-details">
                <h3 className="result-title">
                  {result.prediction ? '🛑 Pneumonia Detected' : '✅ No Pneumonia Detected'}
                </h3>
                <div className="confidence">
                  Confidence: {(result.probability * 100).toFixed(1)}%
                </div>
                <div className="model-used">
                  Best Model: {result.model_name || 'XGBoost'} (Recall: {(result.recall * 100).toFixed(1)}%)
                </div>
                <div className="result-description">
                  {result.prediction ? (
                    <p>This X-ray shows signs of pneumonia. Please consult a healthcare professional immediately.</p>
                  ) : (
                    <p>No signs of pneumonia detected in this X-ray. Regular checkups are still recommended.</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;