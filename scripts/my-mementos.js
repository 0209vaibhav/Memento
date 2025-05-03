// Helper functions for category and tag symbols
function getCategorySymbol(category) {
  const categoryInfo = window.categories.find(c => c.id === category);
  return categoryInfo ? categoryInfo.symbol : '📍';
}

function getTagSymbol(tag) {
  const tagInfo = window.tags.find(t => t.id === tag);
  return tagInfo ? tagInfo.symbol : '🏷️';
}

// Initialize my mementos container
function initializeMyMementos() {
  console.log('Initializing My Mementos container...');
  const myMementosContainer = document.querySelector('.my-mementos-container');
  const myMementosContent = myMementosContainer.querySelector('.my-mementos-content');

  if (!myMementosContainer || !myMementosContent) {
    console.error('Could not find My Mementos container or content elements');
    return;
  }

  console.log('Found My Mementos container elements');

  // Listen for auth state changes
  firebase.auth().onAuthStateChanged(user => {
    console.log('Auth state changed:', user ? 'User logged in' : 'No user');
    if (user) {
      loadUserMementos(user.uid);
    } else {
      myMementosContent.innerHTML = `
        <div class="no-my-mementos-message">
          <p>Please sign in to view your mementos</p>
        </div>
      `;
    }
  });

  // Load mementos for a user
  async function loadUserMementos(userId) {
    console.log('Loading mementos for user:', userId);
    
    // Check if userId is valid
    if (!userId) {
      console.error('Invalid userId provided to loadUserMementos');
      myMementosContent.innerHTML = `
        <div class="error-message">
          <p>Unable to load mementos: User ID is missing.</p>
          <button onclick="window.location.reload()" class="retry-button">
            <i class="fas fa-sync"></i> Retry
          </button>
        </div>
      `;
      return;
    }

    try {
      myMementosContent.innerHTML = '<div class="loading">Loading mementos...</div>';
      
      // First check if the user has access to the mementos collection
      try {
        console.log('Checking Firestore access...');
        await firebase.firestore().collection('mementos').limit(1).get();
        console.log('Firestore access confirmed');
      } catch (error) {
        console.error('Firestore access error:', error);
        if (error.code === 'permission-denied') {
          myMementosContent.innerHTML = `
            <div class="error-message">
              <p>Unable to access mementos. Please make sure you have the correct permissions.</p>
              <button onclick="initializeFirebase()" class="retry-button">
                <i class="fas fa-sync"></i> Initialize Firebase
              </button>
            </div>
          `;
          return;
        }
      }
      
      let hasOrdering = true;
      // Get mementos from Firestore
      let query = firebase.firestore()
        .collection('mementos')
        .where('userId', '==', userId);

      try {
        // Try to add ordering
        query = query.orderBy('timestamp', 'desc');
      } catch (error) {
        console.warn('Timestamp ordering not available yet - index might be building');
        hasOrdering = false;
      }

      console.log('Fetching mementos...');
      const mementosSnapshot = await query.get();
      console.log('Fetched mementos:', mementosSnapshot.size);

      if (mementosSnapshot.empty) {
        myMementosContent.innerHTML = `
          <div class="no-my-mementos-message">
            <p>No mementos found. <br>Start creating new mementos!</p>
          </div>
        `;
        return;
      }

      // Clear the container
      myMementosContent.innerHTML = '';
      
      // Get all mementos and sort them manually if needed
      let mementos = mementosSnapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));

      // Sort manually if we couldn't add ordering to the query
      if (!hasOrdering) {
        mementos.sort((a, b) => {
          const timeA = a.timestamp ? new Date(a.timestamp).getTime() : 0;
          const timeB = b.timestamp ? new Date(b.timestamp).getTime() : 0;
          return timeB - timeA;
        });
      }

      console.log('Rendering mementos...');
      // Render each memento
      for (const memento of mementos) {
        const mementoElement = await createMementoElement(memento);
        myMementosContent.appendChild(mementoElement);
      }
      console.log('Finished rendering mementos');

    } catch (error) {
      console.error('Error loading mementos:', error);
      
      // Check if the error is about missing index
      if (error.message.includes('requires an index')) {
        const indexUrl = error.message.split('create it here: ')[1];
        myMementosContent.innerHTML = `
          <div class="error-message">
            <p>Setting up the database for first use...</p>
            <p>This may take a few minutes. Please try again shortly.</p>
            <button onclick="window.location.reload()" class="retry-button">
              <i class="fas fa-sync"></i> Retry
            </button>
            ${indexUrl ? `
              <p class="admin-note">
                <small>Admin note: Index is being created. 
                <a href="${indexUrl}" target="_blank">Check status here</a></small>
              </p>
            ` : ''}
          </div>
        `;
      } else {
        myMementosContent.innerHTML = `
          <div class="error-message">
            <p>Error loading mementos: ${error.message}</p>
            <button onclick="window.location.reload()" class="retry-button">
              <i class="fas fa-sync"></i> Retry
            </button>
          </div>
        `;
      }
    }
  }

  // Create HTML element for a memento
  async function createMementoElement(memento) {
    // Get username if available
    let username = 'Anonymous';
    if (memento.userId) {
      try {
        const userDoc = await firebase.firestore().collection('users').doc(memento.userId).get();
        if (userDoc.exists) {
          const userData = userDoc.data();
          if (userData.username) {
            username = userData.username;
          } else if (userData.displayName) {
            username = userData.displayName;
          }
        }
      } catch (error) {
        console.error('Error fetching username:', error);
      }
    }

    const mementoElement = document.createElement('div');
    mementoElement.classList.add('memento-item');

    // Handle media content
    let mediaHtml = '<div class="placeholder-media"><i class="fas fa-image"></i></div>';
    if (memento.media && Array.isArray(memento.media) && memento.media.length > 0) {
      const firstMedia = memento.media[0];
      if (typeof firstMedia === 'string') {
        // Check if it's a video by extension
        if (firstMedia.endsWith('.mp4') || firstMedia.endsWith('.webm')) {
          mediaHtml = `<video src="${firstMedia}" controls playsinline></video>`;
        } else {
          mediaHtml = `<img src="${firstMedia}" alt="${memento.name || 'Memento'}" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\\'placeholder-media\\'><i class=\\'fas fa-image\\'></i></div>';">`;
        }
      } else if (firstMedia && typeof firstMedia === 'object') {
        const mediaUrl = firstMedia.url || firstMedia.path || firstMedia.src;
        if (mediaUrl) {
          if (firstMedia.type === 'video' || mediaUrl.endsWith('.mp4') || mediaUrl.endsWith('.webm')) {
            mediaHtml = `<video src="${mediaUrl}" controls playsinline></video>`;
          } else {
            mediaHtml = `<img src="${mediaUrl}" alt="${memento.name || 'Memento'}" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\\'placeholder-media\\'><i class=\\'fas fa-image\\'></i></div>';">`;
          }
        }
      }
    }
    
    // Create the inner HTML structure
    mementoElement.innerHTML = `
      <div class="memento-content">
        <div class="media-section">
          <div class="memento-media">
            ${mediaHtml}
          </div>
          <div class="memento-actions">
            ${memento.location && memento.location.coordinates ? `
              <button class="view-on-map-btn" title="View on map">
                <i class="fas fa-map-marker-alt"></i>
              </button>
            ` : ''}
            <button class="edit-btn" title="Edit memento">
              <i class="fas fa-edit"></i>
            </button>
            <button class="delete-btn" title="Delete memento">
              <i class="fas fa-trash"></i>
            </button>
          </div>
        </div>
        <div class="memento-content">
          <div class="memento-details">
            <h3 class="memento-name">
              <span>${memento.name || 'Untitled Memento'}</span>
            </h3>
            <div class="memento-attributes">
              ${memento.category ? `
                <div class="memento-category">
                  <span class="category-symbol">${getCategorySymbol(memento.category)}</span>
                </div>
              ` : ''}
              ${memento.tags && memento.tags.length > 0 ? `
                <div class="memento-tags">
                  ${memento.tags.slice(0, 3).map(tag => {
                    const tagInfo = window.tags.find(t => t.id === tag) || { symbol: '' };
                    return `<span class="tag">
                      <span class="tag-symbol">${tagInfo.symbol}</span>
                    </span>`;
                  }).join('')}
                  ${memento.tags.length > 3 ? `
                    <span class="tag more-tags">
                      <span class="tag-symbol">+${memento.tags.length - 3}</span>
                    </span>
                  ` : ''}
                </div>
              ` : ''}
            </div>
          </div>
        </div>
      </div>
    `;

    // Add event listeners for buttons
    const viewOnMapBtn = mementoElement.querySelector('.view-on-map-btn');
    if (viewOnMapBtn) {
      viewOnMapBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        try {
          // Get the map element
          const mapElement = document.getElementById('map');
          if (!mapElement) {
            console.warn('Map element not found');
            showToast('Map not found. Please try again.', 'warning');
            return;
          }

          // Get the map instance from the element's data
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
          if (memento.location?.coordinates && 
              !isNaN(memento.location.coordinates.longitude) && 
              !isNaN(memento.location.coordinates.latitude) &&
              memento.location.coordinates.longitude >= -180 &&
              memento.location.coordinates.longitude <= 180 &&
              memento.location.coordinates.latitude >= -90 &&
              memento.location.coordinates.latitude <= 90) {
            
            const center = [
              parseFloat(memento.location.coordinates.longitude),
              parseFloat(memento.location.coordinates.latitude)
            ];
            
            console.log('Flying to coordinates:', center);
            
            mapInstance.flyTo({
              center: center,
              zoom: 22, // Maximum zoom level for closest possible view
              essential: true
            });

            // Find and highlight the corresponding marker
            const markers = window.markers || [];
            const marker = markers.find(m => m.mementoId === memento.id);
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
          } else {
            console.warn('Invalid coordinates for memento:', {
              mementoId: memento.id,
              coordinates: memento.location?.coordinates
            });
            showToast('Invalid location data. Cannot display on map.', 'warning');
          }
        } catch (error) {
          console.error('Error handling map interaction:', error);
          showToast('Error displaying location on map. Please try again.', 'error');
        }
      });
    }

    const editBtn = mementoElement.querySelector('.edit-btn');
    if (editBtn) {
      editBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        editMemento(memento);
      });
    }

    const deleteBtn = mementoElement.querySelector('.delete-btn');
    if (deleteBtn) {
      deleteBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        try {
          const confirmed = await showConfirmationDialog({
            title: 'Delete Memento',
            message: 'Are you sure you want to delete this memento?',
            confirmText: 'Delete',
            cancelText: 'Cancel'
          });

          if (confirmed) {
            await deleteMemento(memento.id);
            showToast('Memento deleted successfully', 'success');
            const user = firebase.auth().currentUser;
            if (user) {
              loadUserMementos(user.uid); // Refresh the list with the current user's ID
            } else {
              console.error('No user logged in after deleting memento');
              showToast('Error refreshing mementos list. Please reload the page.', 'error');
            }
          }
        } catch (error) {
          if (error !== false) { // Only log if it's not a cancellation
            console.error('Error deleting memento:', error);
            showToast('Error deleting memento. Please try again.', 'error');
          }
        }
      });
    }

    // Add click event listener to show memento details
    mementoElement.addEventListener('click', async (e) => {
      e.preventDefault();
      e.stopPropagation();

      // Switch to live feed tab
      const discoveryTab = document.querySelector('.activity-tab[data-activity="discovery"]');
      const liveFeedTab = document.querySelector('.explorer-tab-btn[data-tab="live-feed"]');
      
      if (discoveryTab) {
        discoveryTab.click();
      }
      
      if (liveFeedTab) {
        liveFeedTab.click();
      }

      // Show live feed container
      const liveFeedContainer = document.querySelector('.live-feed-container');
      if (liveFeedContainer) {
        liveFeedContainer.classList.remove('hidden');
      }

      // Switch to memento-info tab and show content
      const mementoInfoTab = document.querySelector('.live-feed-tab-btn[data-tab="memento-info"]');
      const mementoInfoContent = document.getElementById('memento-info-content');
      const mementoInfoList = mementoInfoContent.querySelector('.memento-info-list');

      if (mementoInfoTab) {
        mementoInfoTab.click();
      }

      if (mementoInfoContent) {
        mementoInfoContent.classList.add('active');
      }

      // Create memento HTML...
      if (mementoInfoList) {
        // Format date and time
        const formattedDateTime = new Date(memento.timestamp).toLocaleString('en-US', {
          year: 'numeric',
          month: 'long',
          day: 'numeric',
          hour: 'numeric',
          minute: 'numeric'
        });

        // Create media HTML
        let mediaHtml = `
          <div class="placeholder-media">
            <i class="fas fa-image"></i>
          </div>
        `;

        if (memento.media && memento.media.length > 0) {
          const firstMedia = memento.media[0];
          const mediaUrl = typeof firstMedia === 'string' ? firstMedia : firstMedia.url;
          if (mediaUrl) {
            mediaHtml = `<img src="${mediaUrl}" alt="${memento.name}">`;
          }
        }

        // Create memento HTML
        const mementoInfoHtml = `
          <div class="memento-item">
            <div class="memento-content">
              <div class="media-section">
                <div class="memento-media">
                  ${mediaHtml}
                </div>
                ${memento.location && memento.location.coordinates ? `
                  <div class="memento-actions">
                    <button class="view-on-map-btn" title="View on map">
                      <i class="fas fa-map-marker-alt"></i>
                    </button>
                  </div>
                ` : ''}
              </div>
              <div class="memento-details">
                <h3 class="memento-name">
                  <span>${memento.name || 'Untitled Memento'}</span>
                </h3>
                
                ${memento.description ? `
                  <p class="memento-description">
                    <i class="fas fa-align-left"></i>
                    ${memento.description}
                  </p>
                ` : ''}

                ${memento.location && memento.location.address ? `
                  <p class="memento-location">
                    <i class="fas fa-map-marker-alt"></i>
                    ${memento.location.address}
                  </p>
                ` : ''}

                <p class="memento-timestamp">
                  <i class="fas fa-clock"></i>
                  ${formattedDateTime}
                </p>

                ${memento.duration ? `
                  <p class="memento-duration">
                    <i class="fas fa-hourglass-half"></i>
                    ${window.durations.find(d => d.id === memento.duration)?.name || memento.duration}
                  </p>
                ` : ''}

                ${memento.userId ? `
                  <p class="memento-author">
                    <i class="fas fa-user"></i>
                    ${username}
                  </p>
                ` : ''}
              </div>

              <div class="memento-attributes-footer">
                ${memento.category ? `
                  <div class="memento-category">
                    <span class="category-symbol">${getCategorySymbol(memento.category)}</span>
                    <span class="category-name">${window.categories.find(c => c.id === memento.category)?.name || ''}</span>
                  </div>
                ` : ''}

                ${memento.tags && memento.tags.length > 0 ? `
                  <div class="memento-tags">
                    ${memento.tags.slice(0, 3).map(tag => {
                      const tagInfo = window.tags.find(t => t.id === tag) || { symbol: '' };
                      return `<span class="tag">
                        <span class="tag-symbol">${tagInfo.symbol}</span>
                      </span>`;
                    }).join('')}
                    ${memento.tags.length > 3 ? `
                      <span class="tag more-tags">
                        <span class="tag-symbol">+${memento.tags.length - 3}</span>
                      </span>
                    ` : ''}
                  </div>
                ` : ''}
              </div>
            </div>
          </div>
        `;

        // Update the memento info list content
        mementoInfoList.innerHTML = mementoInfoHtml;

        // Add click handler for the view-on-map button in the info view
        const infoViewMapBtn = mementoInfoList.querySelector('.view-on-map-btn');
        if (infoViewMapBtn) {
          infoViewMapBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            try {
              // Get the map element
              const mapElement = document.getElementById('map');
              if (!mapElement) {
                console.warn('Map element not found');
                showToast('Map not found. Please try again.', 'warning');
                return;
              }

              // Get the map instance from the element's data
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
              console.log('Memento location data:', {
                mementoId: memento.id,
                location: memento.location,
                coordinates: memento.location?.coordinates,
                longitude: memento.location?.coordinates?.longitude,
                latitude: memento.location?.coordinates?.latitude
              });

              if (memento.location?.coordinates && 
                  !isNaN(memento.location.coordinates.longitude) && 
                  !isNaN(memento.location.coordinates.latitude) &&
                  memento.location.coordinates.longitude >= -180 &&
                  memento.location.coordinates.longitude <= 180 &&
                  memento.location.coordinates.latitude >= -90 &&
                  memento.location.coordinates.latitude <= 90) {
                
                const center = [
                  parseFloat(memento.location.coordinates.longitude),
                  parseFloat(memento.location.coordinates.latitude)
                ];
                
                console.log('Flying to coordinates:', center);
                
                mapInstance.flyTo({
                  center: center,
                  zoom: 22, // Maximum zoom level for closest possible view
                  essential: true
                });

                // Find and highlight the corresponding marker
                const markers = window.markers || [];
                const marker = markers.find(m => m.mementoId === memento.id);
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
              } else {
                console.warn('Invalid coordinates for memento:', {
                  mementoId: memento.id,
                  coordinates: memento.location?.coordinates
                });
                showToast('Invalid location data. Cannot display on map.', 'warning');
              }
            } catch (error) {
              console.error('Error handling map interaction:', error);
              showToast('Error displaying location on map. Please try again.', 'error');
            }
          });
        }
      }
    });

    return mementoElement;
  }

  function createMementoInfoContent(memento) {
    // Ensure categories and tags exist and are arrays
    const categories = Array.isArray(memento.categories) ? memento.categories : [];
    const tags = Array.isArray(memento.tags) ? memento.tags : [];

    return `
      <div class="memento-info-item">
        <div class="media-section">
          ${memento.media ? (Array.isArray(memento.media) ? 
            memento.media.map(media => `<img src="${media.url}" alt="Memento media">`).join('') :
            `<img src="${memento.media.url}" alt="Memento media">`) :
            '<div class="media-placeholder"></div>'}
        </div>
        <div class="info-content">
          <h3>${memento.name || 'Untitled Memento'}</h3>
          ${memento.description ? `<p class="description">${memento.description}</p>` : ''}
          <div class="metadata">
            <span class="timestamp">${formatTimestamp(memento.timestamp)}</span>
            ${memento.duration ? `<span class="duration">${memento.duration}</span>` : ''}
          </div>
          ${categories.length > 0 ? `
            <div class="categories">
              ${categories.map(category => getCategorySymbol(category)).join('')}
            </div>
          ` : ''}
          ${tags.length > 0 ? `
            <div class="tags">
              ${tags.map(tag => getTagSymbol(tag)).join('')}
            </div>
          ` : ''}
          ${memento.location ? `
            <div class="location">
              <i class="fas fa-map-marker-alt"></i>
              <span>${memento.location.name || 'Location'}</span>
            </div>
          ` : ''}
        </div>
        <button class="back-to-list-btn">
          <i class="fas fa-arrow-left"></i> Back to List
        </button>
      </div>
    `;
  }

  // Add this function to initialize the back button functionality
  function initializeBackToListButton() {
    document.addEventListener('click', (e) => {
      if (e.target.closest('.back-to-list-btn')) {
        document.querySelector('#my-mementos-info-content').classList.add('hidden');
        document.querySelector('.my-mementos-content').classList.remove('hidden');
      }
    });
  }
}

// Helper function to format date and time for input fields
function formatDateTimeForInput(timestamp) {
  if (!timestamp) return '';
  
  let date;
  try {
    // Handle different timestamp formats
    if (timestamp.toDate) {
      // Firestore Timestamp
      date = timestamp.toDate();
    } else if (typeof timestamp === 'string') {
      // String timestamp
      date = new Date(timestamp);
    } else if (typeof timestamp === 'number') {
      // Numeric timestamp
      date = new Date(timestamp);
    } else {
      // Date object
      date = timestamp;
    }

    // Format to YYYY-MM-DDThh:mm
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');

    return `${year}-${month}-${day}T${hours}:${minutes}`;
  } catch (error) {
    console.error('Error formatting timestamp:', error);
    return '';
  }
}

// Helper function to format date and time for display
function formatDateTime(timestamp) {
  if (!timestamp) return 'No date';
  
  try {
    let date;
    // Handle different timestamp formats
    if (timestamp.toDate) {
      // Firestore Timestamp
      date = timestamp.toDate();
    } else if (typeof timestamp === 'string') {
      // String timestamp
      date = new Date(timestamp);
    } else if (typeof timestamp === 'number') {
      // Numeric timestamp
      date = new Date(timestamp);
    } else {
      // Date object
      date = timestamp;
    }

    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: 'numeric',
      minute: 'numeric'
    });
  } catch (error) {
    console.error('Error formatting timestamp:', error);
    return 'Invalid date';
  }
}

// Make both formatting functions available globally
window.formatDateTime = formatDateTime;
window.formatDateTimeForInput = formatDateTimeForInput;

// Initialize when the document is ready
document.addEventListener('DOMContentLoaded', () => {
  console.log('DOM Content Loaded - initializing My Mementos...');
  initializeMyMementos();
});

// Function to show location on map
function showLocationOnMap(coordinates) {
  try {
    // Get the map element
    const mapElement = document.getElementById('map');
    if (!mapElement) {
      console.warn('Map element not found');
      showToast('Map not found. Please try again.', 'warning');
      return;
    }

    // Get the map instance from the element's data
    const mapInstance = mapElement._mapboxgl_map;
    if (!mapInstance || typeof mapInstance.flyTo !== 'function') {
      console.warn('Map instance not properly initialized');
      showToast('Map is not properly initialized. Please try again.', 'warning');
      return;
    }

    // Collapse the info panel
    const infoTab = document.querySelector('.info-tab');
    const expandLeftBtn = document.getElementById('expand-left');
    const collapseLeftBtn = document.getElementById('collapse-left');
    
    if (infoTab && expandLeftBtn && collapseLeftBtn) {
      infoTab.style.visibility = 'hidden';
      infoTab.classList.add('hidden');
      expandLeftBtn.classList.remove('hidden');
      collapseLeftBtn.classList.add('hidden');
      
      // Resize map after panel collapse
      if (mapInstance) {
        setTimeout(() => { mapInstance.resize(); }, 300);
      }
    }

    // Fly to the location
    mapInstance.flyTo({
      center: [coordinates.longitude, coordinates.latitude],
      zoom: 22, // Maximum zoom level for closest possible view
      essential: true
    });

    // Find and highlight the corresponding marker
    const markers = window.markers || [];
    const marker = markers.find(m => 
      m.getLngLat().lng === coordinates.longitude && 
      m.getLngLat().lat === coordinates.latitude
    );
    
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
    console.error('Error showing location on map:', error);
    showToast('Error displaying location on map. Please try again.', 'error');
  }
}

async function deleteMemento(mementoId) {
  try {
    // Get the current user
    const user = firebase.auth().currentUser;
    if (!user) {
      throw new Error('User not authenticated');
    }

    // Get the memento document reference
    const mementoRef = firebase.firestore().collection('mementos').doc(mementoId);
    const mementoDoc = await mementoRef.get();

    if (!mementoDoc.exists) {
      throw new Error('Memento not found');
    }

    const mementoData = mementoDoc.data();

    // Delete associated media files from Firebase Storage
    if (mementoData.mediaFiles && mementoData.mediaFiles.length > 0) {
      const storage = firebase.storage();
      const deletePromises = mementoData.mediaFiles.map(file => {
        const fileRef = storage.refFromURL(file.url);
        return fileRef.delete();
      });
      await Promise.all(deletePromises);
    }

    // Delete the memento document
    await mementoRef.delete();

    // Remove the marker from the map if it exists
    if (window.markers && mementoData.location && mementoData.location.coordinates) {
      const markerIndex = window.markers.findIndex(marker => {
        const markerCoords = marker.getLngLat().toArray();
        const mementoCoords = [mementoData.location.coordinates.longitude, mementoData.location.coordinates.latitude];
        return markerCoords[0] === mementoCoords[0] && markerCoords[1] === mementoCoords[1];
      });
      
      if (markerIndex !== -1) {
        window.markers[markerIndex].remove();
        window.markers.splice(markerIndex, 1);
      }
    }

    return true;
  } catch (error) {
    console.error('Error in deleteMemento:', error);
    throw error;
  }
} 