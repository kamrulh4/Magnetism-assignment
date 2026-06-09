document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const imageTab = document.getElementById('image-tab');
    const videoTab = document.getElementById('video-tab');
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const dropText = document.getElementById('drop-text');
    const dropHint = document.getElementById('drop-hint');
    const uploadLabel = document.getElementById('upload-label');
    const confSlider = document.getElementById('conf-threshold');
    const confVal = document.getElementById('conf-val');
    const runBtn = document.getElementById('run-btn');
    const visualizer = document.getElementById('visualizer');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingText = document.getElementById('loading-text');
    const viewerHeader = document.getElementById('viewer-header');
    const mediaPlaceholder = document.getElementById('media-placeholder');
    const displayArea = document.getElementById('display-area');
    const downloadBtn = document.getElementById('download-btn');
    const analyticsCard = document.getElementById('analytics-card');
    const totalCount = document.getElementById('total-count');
    const categoriesCount = document.getElementById('categories-count');
    const classBadgeSummary = document.getElementById('class-badge-summary');
    const detectionsTableBody = document.getElementById('detections-table-body');

    // App state
    let activeMode = 'image'; // 'image' or 'video'
    let selectedFile = null;
    let processedImageBase64 = null;
    let processedVideoUrl = null;

    // Slider listener
    confSlider.addEventListener('input', (e) => {
        confVal.textContent = e.target.value;
    });

    // Tab Switching
    imageTab.addEventListener('click', () => switchMode('image'));
    videoTab.addEventListener('click', () => switchMode('video'));

    function switchMode(mode) {
        if (activeMode === mode) return;
        activeMode = mode;
        selectedFile = null;
        processedImageBase64 = null;
        processedVideoUrl = null;

        // UI Reset
        if (mode === 'image') {
            imageTab.classList.add('active');
            videoTab.classList.remove('active');
            uploadLabel.textContent = 'Upload Image';
            dropText.textContent = 'Drag & drop image here';
            dropHint.textContent = 'Supports JPEG, PNG up to 10MB';
            fileInput.setAttribute('accept', 'image/*');
        } else {
            videoTab.classList.add('active');
            imageTab.classList.remove('active');
            uploadLabel.textContent = 'Upload Video';
            dropText.textContent = 'Drag & drop video here';
            dropHint.textContent = 'Supports MP4, WebM up to 50MB';
            fileInput.setAttribute('accept', 'video/*');
        }

        resetResults();
    }

    function resetResults() {
        runBtn.disabled = !selectedFile;
        mediaPlaceholder.style.display = 'flex';
        displayArea.style.display = 'none';
        displayArea.innerHTML = '';
        viewerHeader.style.display = 'none';
        analyticsCard.style.display = 'none';
    }

    // File Selection & Drag/Drop
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    ['dragleave', 'dragend'].forEach(type => {
        dropZone.addEventListener(type, () => {
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        // Validate MIME type
        const isValidImage = file.type.startsWith('image/') && activeMode === 'image';
        const isValidVideo = file.type.startsWith('video/') && activeMode === 'video';

        if (!isValidImage && !isValidVideo) {
            alert(`Please upload a valid ${activeMode} file.`);
            return;
        }

        selectedFile = file;
        resetResults();

        // Update Dropzone visual representation
        dropZone.innerHTML = `
            <div class="file-info">
                <div style="display: flex; align-items: center; gap: 0.5rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                    </svg>
                    <span style="font-weight: 500;">${file.name}</span>
                </div>
                <span style="color: var(--text-muted); font-size: 0.75rem;">${(file.size / (1024 * 1024)).toFixed(2)} MB</span>
            </div>
            <p style="font-size: 0.8rem; color: var(--secondary-color); cursor: pointer; text-decoration: underline;">Change file</p>
        `;

        // Re-bind file selection box in case they click "Change file"
        dropZone.querySelector('p').addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });

        // Show standard preview inside visualizer
        mediaPlaceholder.style.display = 'none';
        displayArea.style.display = 'flex';
        displayArea.innerHTML = '';

        if (activeMode === 'image') {
            const imgEl = document.createElement('img');
            imgEl.src = URL.createObjectURL(file);
            displayArea.appendChild(imgEl);
        } else {
            const videoEl = document.createElement('video');
            videoEl.src = URL.createObjectURL(file);
            videoEl.controls = true;
            videoEl.autoplay = false;
            displayArea.appendChild(videoEl);
        }
    }

    // Run Detection
    runBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Show Loading Overlay
        loadingText.textContent = `Processing ${activeMode} with YOLOv8n... please wait`;
        loadingOverlay.style.display = 'flex';
        runBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('conf', confSlider.value);

        try {
            if (activeMode === 'image') {
                const response = await fetch('/detect-image/', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`Server returned error: ${response.statusText}`);
                }

                const data = await response.json();
                processedImageBase64 = data.annotated_image;

                // Render image
                displayArea.innerHTML = `<img src="${processedImageBase64}" alt="Annotated Image">`;
                viewerHeader.style.display = 'flex';

                // Display Analytics
                renderAnalytics(data.detections);

            } else {
                // Video processing
                const response = await fetch('/detect-video/', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`Server returned error: ${response.statusText}`);
                }

                const blob = await response.blob();
                processedVideoUrl = URL.createObjectURL(blob);

                // Render video
                displayArea.innerHTML = `
                    <video controls autoplay loop>
                        <source src="${processedVideoUrl}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                `;
                viewerHeader.style.display = 'flex';

                // Video Analytics
                renderVideoAnalytics();
            }
        } catch (error) {
            console.error('Detection Error:', error);
            alert(`An error occurred during detection: ${error.message}`);
            resetResults();
        } finally {
            loadingOverlay.style.display = 'none';
            runBtn.disabled = false;
        }
    });

    // Image Download
    downloadBtn.addEventListener('click', () => {
        if (activeMode === 'image' && processedImageBase64) {
            const link = document.createElement('a');
            link.href = processedImageBase64;
            link.download = `annotated_${selectedFile.name}`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } else if (activeMode === 'video' && processedVideoUrl) {
            const link = document.createElement('a');
            link.href = processedVideoUrl;
            link.download = `annotated_${selectedFile.name}`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    });

    // Render Analytics Dashboard (Image mode)
    function renderAnalytics(detections) {
        analyticsCard.style.display = 'flex';
        totalCount.textContent = detections.length;

        // Group categories
        const categories = {};
        detections.forEach(det => {
            const cat = det.class_name;
            categories[cat] = (categories[cat] || 0) + 1;
        });

        categoriesCount.textContent = Object.keys(categories).length;

        // Badges Render
        classBadgeSummary.innerHTML = '';
        if (detections.length === 0) {
            classBadgeSummary.innerHTML = '<span style="color: var(--text-muted); font-size: 0.85rem;">No objects detected</span>';
        } else {
            Object.entries(categories).forEach(([name, count]) => {
                const badge = document.createElement('div');
                badge.className = 'class-badge';
                badge.innerHTML = `${name} <span>${count}</span>`;
                classBadgeSummary.appendChild(badge);
            });
        }

        // Table Render
        detectionsTableBody.innerHTML = '';
        if (detections.length === 0) {
            detectionsTableBody.innerHTML = '<tr><td colspan="3" style="text-align: center; color: var(--text-muted);">No detections</td></tr>';
        } else {
            detections.forEach(det => {
                const tr = document.createElement('tr');
                
                // Class name
                const tdClass = document.createElement('td');
                tdClass.style.fontWeight = '600';
                tdClass.textContent = det.class_name;
                
                // Confidence badge
                const tdConf = document.createElement('td');
                const confPercentage = (det.confidence * 100).toFixed(1) + '%';
                let confClass = 'conf-low';
                if (det.confidence >= 0.75) confClass = 'conf-high';
                else if (det.confidence >= 0.40) confClass = 'conf-mid';
                tdConf.innerHTML = `<span class="conf-pill ${confClass}">${confPercentage}</span>`;

                // BBox coordinates
                const tdBbox = document.createElement('td');
                tdBbox.style.fontFamily = 'monospace';
                tdBbox.style.color = 'var(--text-muted)';
                tdBbox.textContent = `[${det.bbox.join(', ')}]`;

                tr.appendChild(tdClass);
                tr.appendChild(tdConf);
                tr.appendChild(tdBbox);
                detectionsTableBody.appendChild(tr);
            });
        }
    }

    // Video Analytics Render
    function renderVideoAnalytics() {
        analyticsCard.style.display = 'flex';
        totalCount.textContent = '-';
        categoriesCount.textContent = '-';
        classBadgeSummary.innerHTML = '<span style="color: var(--secondary-color); font-size: 0.85rem; font-weight: 500;">✓ Video processed successfully and annotated with YOLOv8n. Play/download video to review.</span>';
        detectionsTableBody.innerHTML = '<tr><td colspan="3" style="text-align: center; color: var(--text-muted);">Detailed frame telemetry is compiled inside the video.</td></tr>';
    }
});
