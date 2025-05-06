// Format timestamp to readable date
function formatTimestamp(timestamp) {
    if (!timestamp) return 'No date';
    
    try {
        // Convert Firebase timestamp to Date
        const date = timestamp.toDate ? timestamp.toDate() : new Date(timestamp);
        
        // Format the date
        return new Intl.DateTimeFormat('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        }).format(date);
    } catch (error) {
        console.error('Error formatting timestamp:', error);
        return 'Invalid date';
    }
}

// Initialize drafts container
function initializeDraftsContainer() {
  const draftsContainer = document.getElementById('journey-drafts-container');
  const draftsContent = draftsContainer.querySelector('.drafts-content');

  // Listen for auth state changes
  firebase.auth().onAuthStateChanged(user => {
    if (user) {
      loadDrafts(user.uid);
    } else {
      draftsContent.innerHTML = `
        <div class="no-drafts-message">
          <p>Please sign in to view your drafts</p>
        </div>
      `;
    }
  });

  // Load drafts for a user
  async function loadDrafts(userId) {
    try {
      draftsContent.innerHTML = '<div class="loading">Loading drafts...</div>';
      
      // Get drafts without ordering first
      const draftsSnapshot = await firebase.firestore()
        .collection('memento_drafts')
        .where('userId', '==', userId)
        .get();

      if (draftsSnapshot.empty) {
        draftsContent.innerHTML = `
          <div class="no-drafts-message">
            <p>No drafts found. <br>Start creating new mementos!</p>
          </div>
        `;
        return;
      }

      // Render drafts
      draftsContent.innerHTML = '';
      
      // Convert to array and sort manually
      const drafts = [];
      draftsSnapshot.forEach(doc => {
        drafts.push({
          id: doc.id,
          ...doc.data()
        });
      });

      // Sort by timestamp if exists, otherwise by id
      drafts.sort((a, b) => {
        if (a.timestamp && b.timestamp) {
          return b.timestamp - a.timestamp;
        }
        return a.id > b.id ? -1 : 1;
      });

      // Render sorted drafts
      drafts.forEach(draft => {
        const draftElement = createDraftElement(draft);
        draftsContent.appendChild(draftElement);
      });

    } catch (error) {
      console.error('Error loading drafts:', error);
      draftsContent.innerHTML = `
        <div class="error-message">
          <p>Error loading drafts: ${error.message}</p>
        </div>
      `;
    }
  }

  // Create HTML element for a draft
  function createDraftElement(draft) {
    const draftElement = document.createElement('div');
    draftElement.className = 'draft-item';

    // Handle media content
    let mediaHtml = '<div class="placeholder-media"><i class="fas fa-image"></i></div>';
    if (draft.media && Array.isArray(draft.media) && draft.media.length > 0) {
        const firstMedia = draft.media[0];
        if (typeof firstMedia === 'string') {
            // Check if it's a video by extension
            if (firstMedia.endsWith('.mp4') || firstMedia.endsWith('.webm')) {
                mediaHtml = `<video src="${firstMedia}" controls playsinline></video>`;
            } else {
            mediaHtml = `<img src="${firstMedia}" alt="${draft.name || 'Draft'}" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\\'placeholder-media\\'><i class=\\'fas fa-image\\'></i></div>';">`;
            }
        } else if (firstMedia && typeof firstMedia === 'object') {
            const mediaUrl = firstMedia.url || firstMedia.path || firstMedia.src;
            if (mediaUrl) {
                if (firstMedia.type === 'video' || mediaUrl.endsWith('.mp4') || mediaUrl.endsWith('.webm')) {
                    mediaHtml = `<video src="${mediaUrl}" controls playsinline></video>`;
                } else {
                    mediaHtml = `<img src="${mediaUrl}" alt="${draft.name || 'Draft'}" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\\'placeholder-media\\'><i class=\\'fas fa-image\\'></i></div>';">`;
                }
            }
        }
    }

    // Create the inner HTML structure
    draftElement.innerHTML = `
        <div class="memento-content">
            <div class="media-section">
                <div class="memento-media">
                    ${mediaHtml}
                </div>
                <div class="memento-actions">
                    ${draft.location && draft.location.coordinates ? `
                        <button class="view-on-map-btn" title="View on map">
                            <i class="fas fa-map-marker-alt"></i>
                        </button>
                    ` : ''}
                    <button class="edit-btn" title="Edit draft">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="delete-btn" title="Delete draft">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
            <div class="memento-content">
                <div class="memento-details">
                    <h3 class="memento-name">
                        <span>${draft.name || 'Untitled Draft'}</span>
                    </h3>
                    <div class="memento-attributes">
                        ${draft.category ? `
                            <span class="memento-category">
                                ${getCategorySymbol(draft.category)}
                            </span>
                        ` : ''}
                        ${draft.tags && draft.tags.length > 0 ? `
                            <span class="memento-tags">
                                ${draft.tags.map(tag => {
                                    const tagInfo = window.tags.find(t => t.id === tag) || { symbol: '' };
                                    return `<span class="tag">${tagInfo.symbol}</span>`;
                                }).join('')}
                            </span>
                        ` : ''}
                    </div>
                </div>
            </div>
        </div>
    `;

    // Add event listeners for buttons
    const viewOnMapBtn = draftElement.querySelector('.view-on-map-btn');
    if (viewOnMapBtn) {
        viewOnMapBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            try {
                // Get the map element
                const mapElement = document.getElementById('map');
                if (!mapElement) {
                    console.warn('Map element not found');
                    showToast('Map not found. Please try again.', 'warning');
                    return;
                }

                // Get the map instance
                const mapInstance = mapElement._mapboxgl_map;
                if (!mapInstance || typeof mapInstance.flyTo !== 'function') {
                    console.warn('Map instance not properly initialized');
                    showToast('Map is not properly initialized. Please try again.', 'warning');
                    return;
                }

                // On mobile devices (width < 1024px), collapse the info panel
                if (window.innerWidth < 1024) {
                    const infoTab = document.querySelector('.info-tab');
                    const expandLeftBtn = document.getElementById('expand-left');
                    const collapseLeftBtn = document.getElementById('collapse-left');
                    
                    if (infoTab && expandLeftBtn && collapseLeftBtn) {
                        infoTab.style.visibility = 'hidden';
                        infoTab.classList.add('hidden');
                        expandLeftBtn.classList.remove('hidden');
                        collapseLeftBtn.classList.add('hidden');
                        
                        // Resize map after panel is hidden
                        setTimeout(() => {
                            if (mapInstance) {
                                mapInstance.resize();
                            }
                        }, 300);
                    }
                }

                // Fly to the location
                mapInstance.flyTo({
                    center: [draft.location.coordinates.longitude, draft.location.coordinates.latitude],
                    zoom: 22, // Maximum zoom level for closest possible view
                    essential: true
                });

                // Find and highlight the corresponding marker
                const markers = window.markers || [];
                const marker = markers.find(m => m.mementoId === draft.id);
                if (marker) {
                    // Remove highlight from all markers
                    markers.forEach(m => {
                        if (m.element) {
                            m.element.classList.remove('highlighted-marker');
                        }
                    });
                    // Add highlight to the selected marker
                    if (marker.element) {
                        marker.element.classList.add('highlighted-marker');
                    }
                    // Show popup
                    marker.togglePopup();
                }
            } catch (error) {
                console.error('Error handling map interaction:', error);
                showToast('Error displaying location on map. Please try again.', 'error');
            }
        });
    }

    const editBtn = draftElement.querySelector('.edit-btn');
    if (editBtn) {
        editBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            editDraft(draft);
        });
    }

    const deleteBtn = draftElement.querySelector('.delete-btn');
    if (deleteBtn) {
        deleteBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            try {
                const confirmed = await showConfirmationDialog({
                    title: 'Delete Draft',
                    message: 'Are you sure you want to delete this draft?',
                    confirmText: 'Delete',
                    cancelText: 'Cancel'
                });

                if (confirmed) {
                    await deleteDraft(draft.id);
                    draftElement.remove();
                    showToast('Draft deleted successfully', 'success');
                }
            } catch (error) {
                if (error !== false) { // Only log if it's not a cancellation
                    console.error('Error deleting draft:', error);
                    showToast('Failed to delete draft. Please try again.', 'error');
                }
            }
        });
    }

    // Add click event listener to show draft details
    draftElement.addEventListener('click', () => {
        editDraft(draft);
    });

    return draftElement;
  }

  // Helper function to get category symbol
  function getCategorySymbol(categoryId) {
    const category = window.categories.find(c => c.id === categoryId);
    return category ? category.symbol : '🗂️';
  }

  // Helper function to get tag symbol
  function getTagSymbol(tagId) {
    const tag = window.tags.find(t => t.id === tagId);
    return tag ? tag.symbol : '🏷️';
  }

  // Edit draft
  async function editDraft(draft) {
    try {
      // Show the capture form
      const captureForm = document.getElementById('journey-capture-form');
      const journeyCaptureForm = document.getElementById('journey-capture-form');
      
      if (!journeyCaptureForm) {
        throw new Error('Capture form not found');
      }

      // Update preview elements
      const namePreview = journeyCaptureForm.querySelector('#name-preview');
      const descriptionPreview = journeyCaptureForm.querySelector('#description-preview');
      const tagsPreview = journeyCaptureForm.querySelector('#tags-preview');
      const categoryPreview = journeyCaptureForm.querySelector('#category-preview');
      const locationPreview = journeyCaptureForm.querySelector('#location-preview');
      const timestampPreview = journeyCaptureForm.querySelector('#timestamp-preview');
      const durationPreview = journeyCaptureForm.querySelector('#duration-preview');
      const mediaPreview = journeyCaptureForm.querySelector('#media-preview');

      // Populate the form with draft data
      if (namePreview) namePreview.textContent = draft.name || 'Add name';
      if (descriptionPreview) descriptionPreview.textContent = draft.description || 'Add description';
      if (tagsPreview) tagsPreview.textContent = draft.tags?.length ? `${draft.tags.length} tags selected` : 'Select tags';
      if (categoryPreview) categoryPreview.textContent = draft.category || 'Select category';
      if (locationPreview) locationPreview.textContent = draft.location?.address || 'Add location';
      if (timestampPreview) timestampPreview.textContent = draft.timestamp ? new Date(draft.timestamp).toLocaleString() : 'Current time';
      if (durationPreview) durationPreview.textContent = draft.duration || 'Select duration';
      if (mediaPreview) {
        mediaPreview.textContent = draft.media && draft.media.length > 0 
          ? `${draft.media.length} media items selected` 
          : 'Add photo/video';
      }

      // Store the draft ID for later use
      captureForm.dataset.draftId = draft.id;

      // Show capture form and hide drafts
      captureForm.classList.remove('hidden');
      draftsContainer.classList.add('hidden');

    } catch (error) {
      console.error('Error editing draft:', error);
      showToast(`Error editing draft: ${error.message}`, 'error');
    }
  }

  // Delete draft
  async function deleteDraft(draftId) {
    try {
      await firebase.firestore().collection('memento_drafts').doc(draftId).delete();
      
      // If no more drafts, show empty message
      if (draftsContent.children.length === 0) {
        draftsContent.innerHTML = `
          <div class="no-drafts-message">
            <p>No drafts found. Start creating new mementos!</p>
          </div>
        `;
      }
    } catch (error) {
      console.error('Error deleting draft:', error);
      showToast('Error deleting draft', 'error');
    }
  }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeDraftsContainer);
