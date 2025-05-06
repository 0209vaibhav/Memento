// Initialize curated functionality
function initializeCurated() {
  console.log('Initializing Curated container...');
  
  // Get DOM elements
  const curatedContainer = document.querySelector('.curated-container');
  const curatedTabs = document.querySelectorAll('.curated-tab-btn');
  const curatedContents = document.querySelectorAll('.curated-tab-content');
  
  // Add click event listeners to tabs
  curatedTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      // Remove active class from all tabs and contents
      curatedTabs.forEach(t => t.classList.remove('active'));
      curatedContents.forEach(c => c.classList.remove('active'));
      
      // Add active class to clicked tab
      tab.classList.add('active');
      
      // Show corresponding content
      const tabId = tab.getAttribute('data-tab');
      const content = document.getElementById(`${tabId}-content`);
      if (content) {
        content.classList.add('active');
      }
      
      // Load content based on tab
      switch(tabId) {
        case 'trending':
          loadTrendingContent();
          break;
        case 'recommended':
          loadRecommendedContent();
          break;
        case 'daily':
          loadDailyContent();
          break;
        case 'near-me':
          loadNearMeContent();
          break;
      }
    });
  });
  
  // Load initial content (trending)
  loadTrendingContent();
}

// Function to handle scroll-based pagination highlighting
function setupScrollBasedPagination(listElement, paginationContainer) {
  if (!listElement || !paginationContainer) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        // Find the index of the memento that is currently in view
        const mementoIndex = Array.from(listElement.children).indexOf(entry.target);
        if (mementoIndex !== -1) {
          // Remove active class from all buttons
          paginationContainer.querySelectorAll('.pagination-btn').forEach(btn => {
            btn.classList.remove('active');
          });
          // Add active class to the corresponding button
          const button = paginationContainer.querySelector(`.pagination-btn:nth-child(${mementoIndex + 1})`);
          if (button) {
            button.classList.add('active');
          }
        }
      }
    });
  }, {
    threshold: 0.5, // Trigger when 50% of the memento is visible
    rootMargin: '-10% 0px' // Add a small margin to make the detection more precise
  });

  // Observe all memento elements
  listElement.querySelectorAll('.memento-item').forEach(memento => {
    observer.observe(memento);
  });
}

// Load trending content
async function loadTrendingContent() {
  try {
    const trendingList = document.querySelector('#trending-content .curated-list');
    if (!trendingList) return;
    
    // Show loading state
    trendingList.innerHTML = `
      <div class="trending-message">
        <i class="fas fa-spinner fa-spin"></i>
        <p>Loading trending mementos...</p>
      </div>
    `;
    
    // Get all mementos using the existing function
    const allMementos = await getAllMementos();
    
    // Clear existing content
    trendingList.innerHTML = '';
    
    // Remove any existing pagination
    const existingPagination = trendingList.parentElement.querySelector('.trending-pagination');
    if (existingPagination) {
      existingPagination.remove();
    }
    
    // Calculate trending score for each memento
    const mementos = allMementos.map(memento => {
      // Get current time
      const now = new Date();
      
      // Get memento time
      const mementoTime = memento.timestamp ? new Date(memento.timestamp) : new Date();
      
      // Calculate time difference in hours
      const hoursDiff = (now - mementoTime) / (1000 * 60 * 60);
      
      // Calculate trending score
      // Formula: (likes * 1 + favorites * 2) / (hours + 2)^1.5
      // This gives more weight to recent mementos and favorites
      const likes = memento.likeCount || 0;
      const favorites = memento.favoriteCount || 0;
      const trendingScore = (likes * 1 + favorites * 2) / Math.pow(hoursDiff + 2, 1.5);
      
      return {
        ...memento,
        trendingScore
      };
    }).sort((a, b) => b.trendingScore - a.trendingScore)
      .slice(0, 5); // Always show top 5 trending mementos
    
    // Create pagination container
    const paginationContainer = document.createElement('div');
    paginationContainer.className = 'trending-pagination';
    
    // Create pagination buttons
    for (let i = 0; i < mementos.length; i++) {
      const button = document.createElement('button');
      button.className = 'pagination-btn';
      button.textContent = i + 1;
      button.title = `View trending memento #${i + 1}`;
      
      // Add click handler
      button.addEventListener('click', () => {
        // Remove active class from all buttons
        paginationContainer.querySelectorAll('.pagination-btn').forEach(btn => {
          btn.classList.remove('active');
        });
        
        // Add active class to clicked button
        button.classList.add('active');
        
        // Scroll to the corresponding memento
        const mementoElement = trendingList.children[i];
        if (mementoElement) {
          mementoElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
      
      paginationContainer.appendChild(button);
    }

    // Add refresh button to pagination container
    const refreshBtn = document.createElement('button');
    refreshBtn.className = 'refresh-btn';
    refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i>';
    refreshBtn.title = 'Refresh trending mementos';
    refreshBtn.addEventListener('click', (e) => {
      e.preventDefault();
      refreshTrendingContent();
    });
    paginationContainer.appendChild(refreshBtn);
    
    // Add pagination container before the list
    trendingList.parentElement.insertBefore(paginationContainer, trendingList);
    
    // Add each memento to the list
    for (const memento of mementos) {
      const mementoElement = await createCuratedMementoElement(memento);
      trendingList.appendChild(mementoElement);
    }

    // Setup scroll-based pagination
    setupScrollBasedPagination(trendingList, paginationContainer);

    // If no mementos found, show a message but keep the pagination
    if (mementos.length === 0) {
      trendingList.innerHTML = `
        <div class="trending-message">
          <i class="fas fa-fire"></i>
          <p>No trending mementos yet.<br>Be the first to create one!</p>
        </div>
      `;
    }
  } catch (error) {
    console.error('Error loading trending content:', error);
    showToast('Error loading trending content', 'error');
  }
}

// Refresh trending content
async function refreshTrendingContent() {
  try {
    const trendingList = document.querySelector('#trending-content .curated-list');
    if (!trendingList) return;
    
    // Show loading state
    trendingList.innerHTML = `
      <div class="trending-message">
        <i class="fas fa-spinner fa-spin"></i>
        <p>Refreshing trending mementos...</p>
      </div>
    `;
    
    // Reload trending content
    await loadTrendingContent();
    
    // Show success message
    showToast('Trending mementos refreshed', 'success');
  } catch (error) {
    console.error('Error refreshing trending content:', error);
    showToast('Error refreshing trending content', 'error');
  }
}

// Load recommended content
async function loadRecommendedContent() {
  try {
    const recommendedList = document.querySelector('#recommended-content .curated-list');
    if (!recommendedList) return;
    
    // Get current user
    const user = firebase.auth().currentUser;
    if (!user) {
      recommendedList.innerHTML = `
        <div class="recommended-message">
          <i class="fas fa-user-lock"></i>
          <p>Please sign in to see personalized recommendations</p>
        </div>
      `;
      return;
    }
    
    // Get user's profile data
    const userDoc = await firebase.firestore()
      .collection('users')
      .doc(user.uid)
      .get();
    
    const userData = userDoc.data();
    
    // Get all mementos using the existing function
    const allMementos = await getAllMementos();
    
    // Clear existing content
    recommendedList.innerHTML = '';
    
    // Remove any existing pagination
    const existingPagination = recommendedList.parentElement.querySelector('.recommendation-pagination');
    if (existingPagination) {
      existingPagination.remove();
    }
    
    // Calculate recommendation scores for each memento
    const scoredMementos = allMementos.map(memento => {
      let score = 0;
      
      // 1. Category matching (30% weight)
      if (userData?.preferredCategories?.includes(memento.category)) {
        score += 30;
      }
      
      // 2. Tag matching (20% weight)
      if (memento.tags && userData?.preferredTags) {
        const matchingTags = memento.tags.filter(tag => 
          userData.preferredTags.includes(tag)
        );
        score += (matchingTags.length / memento.tags.length) * 20;
      }
      
      // 3. Location proximity (20% weight)
      if (memento.location && userData?.location) {
        const distance = calculateDistance(
          memento.location.coordinates,
          userData.location.coordinates
        );
        // Score decreases as distance increases
        score += Math.max(0, 20 - (distance / 5));
      }
      
      // 4. Time relevance (15% weight)
      const mementoTime = new Date(memento.timestamp);
      const now = new Date();
      const hoursDiff = (now - mementoTime) / (1000 * 60 * 60);
      score += Math.max(0, 15 - (hoursDiff / 24));
      
      // 5. Popularity (15% weight)
      const likes = memento.likeCount || 0;
      const favorites = memento.favoriteCount || 0;
      score += Math.min(15, (likes + favorites * 2) / 10);
      
      return {
        ...memento,
        recommendationScore: score
      };
    });
    
    // Sort by recommendation score
    let recommendedMementos = scoredMementos
      .sort((a, b) => b.recommendationScore - a.recommendationScore)
      .slice(0, 5); // Show top 5 recommendations
    
    // If no personalized recommendations, use fallback strategies
    if (recommendedMementos.length === 0 || recommendedMementos.every(m => m.recommendationScore === 0)) {
      // Fallback 1: Recent popular mementos
      recommendedMementos = allMementos
        .sort((a, b) => {
          const aScore = (a.likeCount || 0) + (a.favoriteCount || 0) * 2;
          const bScore = (b.likeCount || 0) + (b.favoriteCount || 0) * 2;
          return bScore - aScore;
        })
        .slice(0, 5);
      
      // If still empty, Fallback 2: Most recent mementos
      if (recommendedMementos.length === 0) {
        recommendedMementos = allMementos
          .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
          .slice(0, 5);
      }
      
      // If still empty, Fallback 3: Random mementos
      if (recommendedMementos.length === 0) {
        recommendedMementos = allMementos
          .sort(() => Math.random() - 0.5)
          .slice(0, 5);
      }
    }

    // Create pagination container
    const paginationContainer = document.createElement('div');
    paginationContainer.className = 'recommendation-pagination';
    
    // Create pagination buttons
    for (let i = 0; i < recommendedMementos.length; i++) {
      const button = document.createElement('button');
      button.className = 'pagination-btn';
      button.textContent = i + 1;
      button.title = `View recommendation #${i + 1}`;
      
      // Add click handler
      button.addEventListener('click', () => {
        // Remove active class from all buttons
        paginationContainer.querySelectorAll('.pagination-btn').forEach(btn => {
          btn.classList.remove('active');
        });
        
        // Add active class to clicked button
        button.classList.add('active');
        
        // Scroll to the corresponding memento
        const mementoElement = recommendedList.children[i];
        if (mementoElement) {
          mementoElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
      
      paginationContainer.appendChild(button);
    }

    // Add refresh button to pagination container
    const refreshBtn = document.createElement('button');
    refreshBtn.className = 'refresh-btn';
    refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i>';
    refreshBtn.title = 'Refresh recommendations';
    refreshBtn.addEventListener('click', (e) => {
      e.preventDefault();
      refreshRecommendations();
    });
    paginationContainer.appendChild(refreshBtn);
    
    // Add pagination container before the list
    recommendedList.parentElement.insertBefore(paginationContainer, recommendedList);
    
    // Add each memento to the list
    for (const memento of recommendedMementos) {
      const mementoElement = await createCuratedMementoElement(memento);
      recommendedList.appendChild(mementoElement);
    }

    // Setup scroll-based pagination
    setupScrollBasedPagination(recommendedList, paginationContainer);
  } catch (error) {
    console.error('Error loading recommended content:', error);
    showToast('Error loading recommended content', 'error');
  }
}

// Helper function to calculate distance between two points
function calculateDistance(point1, point2) {
  const R = 3959; // Earth's radius in miles
  const lat1 = point1.latitude * Math.PI / 180;
  const lat2 = point2.latitude * Math.PI / 180;
  const dLat = (point2.latitude - point1.latitude) * Math.PI / 180;
  const dLon = (point2.longitude - point1.longitude) * Math.PI / 180;

  const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
          Math.cos(lat1) * Math.cos(lat2) *
          Math.sin(dLon/2) * Math.sin(dLon/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  return R * c;
}

// Load daily content
async function loadDailyContent() {
  try {
    const dailyList = document.querySelector('#daily-content .curated-list');
    if (!dailyList) return;
    
    // Show loading state
    dailyList.innerHTML = `
      <div class="daily-message">
        <i class="fas fa-spinner fa-spin"></i>
        <p>Loading daily mementos...</p>
      </div>
    `;
    
    // Get all mementos using the existing function
    const allMementos = await getAllMementos();
    
    // Clear existing content
    dailyList.innerHTML = '';
    
    // Remove any existing pagination
    const existingPagination = dailyList.parentElement.querySelector('.daily-pagination');
    if (existingPagination) {
      existingPagination.remove();
    }
    
    // Get current time and 24 hours ago
    const now = new Date();
    const twentyFourHoursAgo = new Date(now.getTime() - (24 * 60 * 60 * 1000));
    
    // Filter and sort mementos
    const mementos = allMementos
      .filter(memento => {
        const mementoDate = new Date(memento.timestamp);
        return mementoDate >= twentyFourHoursAgo;
      })
      .sort((a, b) => {
        // First sort by likes and favorites
        const aScore = (a.likeCount || 0) + (a.favoriteCount || 0) * 2;
        const bScore = (b.likeCount || 0) + (b.favoriteCount || 0) * 2;
        if (bScore !== aScore) return bScore - aScore;
        
        // If scores are equal, sort by timestamp (most recent first)
        return new Date(b.timestamp) - new Date(a.timestamp);
      })
      .slice(0, 5); // Show top 5 daily mementos
    
    // Create pagination container
    const paginationContainer = document.createElement('div');
    paginationContainer.className = 'daily-pagination';
    
    // Create pagination buttons
    for (let i = 0; i < mementos.length; i++) {
      const button = document.createElement('button');
      button.className = 'pagination-btn';
      button.textContent = i + 1;
      button.title = `View daily memento #${i + 1}`;
      
      // Add click handler
      button.addEventListener('click', () => {
        // Remove active class from all buttons
        paginationContainer.querySelectorAll('.pagination-btn').forEach(btn => {
          btn.classList.remove('active');
        });
        
        // Add active class to clicked button
        button.classList.add('active');
        
        // Scroll to the corresponding memento
        const mementoElement = dailyList.children[i];
        if (mementoElement) {
          mementoElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
      
      paginationContainer.appendChild(button);
    }

    // Add refresh button to pagination container
    const refreshBtn = document.createElement('button');
    refreshBtn.className = 'refresh-btn';
    refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i>';
    refreshBtn.title = 'Refresh daily mementos';
    refreshBtn.addEventListener('click', (e) => {
      e.preventDefault();
      refreshDailyContent();
    });
    paginationContainer.appendChild(refreshBtn);
    
    // Add pagination container before the list
    dailyList.parentElement.insertBefore(paginationContainer, dailyList);
    
    // Add each memento to the list
    for (const memento of mementos) {
      const mementoElement = await createCuratedMementoElement(memento);
      dailyList.appendChild(mementoElement);
    }

    // Setup scroll-based pagination
    setupScrollBasedPagination(dailyList, paginationContainer);

    // If no mementos found, show a message but keep the pagination
    if (mementos.length === 0) {
      dailyList.innerHTML = `
        <div class="daily-message">
          <i class="fas fa-calendar-day"></i>
          <p>No mementos in the last 24 hours.<br>Be the first to create one!</p>
        </div>
      `;
    }
  } catch (error) {
    console.error('Error loading daily content:', error);
    showToast('Error loading daily content', 'error');
  }
}

// Refresh daily content
async function refreshDailyContent() {
  try {
    const dailyList = document.querySelector('#daily-content .curated-list');
    if (!dailyList) return;
    
    // Show loading state
    dailyList.innerHTML = `
      <div class="daily-message">
        <i class="fas fa-spinner fa-spin"></i>
        <p>Refreshing daily mementos...</p>
      </div>
    `;
    
    // Reload daily content
    await loadDailyContent();
    
    // Show success message
    showToast('Daily mementos refreshed', 'success');
  } catch (error) {
    console.error('Error refreshing daily content:', error);
    showToast('Error refreshing daily content', 'error');
  }
}

// Create curated memento element
async function createCuratedMementoElement(memento) {
  const mementoElement = document.createElement('div');
  mementoElement.className = 'memento-item';
  
  // Format date and time
  const formattedDateTime = new Date(memento.timestamp).toLocaleString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: 'numeric',
    minute: 'numeric'
  });
  
  // Get username if available
  let username = 'Anonymous';
  if (memento.userId) {
    // Check if this is a public memento (userId contains spaces or slashes)
    if (memento.userId.includes('/') || memento.userId.includes(' ')) {
      // For public mementos, use the userId directly as the source/author
      username = memento.userId;
    } else {
      // For user mementos, try to fetch from Firebase
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
        // Use userId as fallback if Firebase query fails
        username = memento.userId;
      }
    }
  }
  
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
  
  // Create memento HTML
  mementoElement.innerHTML = `
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
            <span class="memento-description-text">
              ${memento.description}${memento.link ? `<a href="${memento.link}" target="_blank" class="view-more-link">view more</a>` : ''}
            </span>
          </p>
        ` : ''}

        ${memento.location && memento.location.address ? `
          <p class="memento-location">
            <i class="fas fa-map-marker-alt"></i>
            ${memento.location.address}
          </p>
        ` : ''}

        ${memento.timestamp ? `
          <p class="memento-timestamp">
            <i class="fas fa-clock"></i>
            ${formattedDateTime}
          </p>
        ` : ''}

        ${memento.duration ? `
          <p class="memento-duration">
            <i class="fas fa-hourglass-half"></i>
            <span class="memento-duration-text">${memento.duration}</span>
          </p>
        ` : ''}

        ${memento.userId ? `
          <p class="memento-author">
            <i class="fas fa-user"></i>
            <span class="memento-author-text">${username}</span>
          </p>
        ` : ''}

        <div class="memento-engagement">
          <button class="like-btn ${memento.isLiked ? 'liked' : ''}" data-memento-id="${memento.id}">
            <i class="fas fa-heart"></i>
            <span class="like-count">${memento.likeCount || 0}</span>
          </button>
          <button class="favorite-btn ${memento.isFavorited ? 'favorited' : ''}" data-memento-id="${memento.id}">
            <i class="fas fa-bookmark"></i>
            <span class="favorite-count">${memento.favoriteCount || 0}</span>
          </button>
        </div>
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
            ${memento.tags.map(tag => {
              const tagInfo = window.tags.find(t => t.id === tag) || { symbol: '', name: '' };
              return `<span class="tag">
                <span class="tag-symbol">${tagInfo.symbol}</span>
                <span class="tag-name">${tagInfo.name}</span>
              </span>`;
            }).join('')}
          </div>
        ` : ''}
      </div>
    </div>
  `;
  
  // Add click handler for the view-on-map button
  const viewOnMapBtn = mementoElement.querySelector('.view-on-map-btn');
  if (viewOnMapBtn && memento.location && memento.location.coordinates) {
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
        mapInstance.flyTo({
          center: [memento.location.coordinates.longitude, memento.location.coordinates.latitude],
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

        // Always show live feed info list
        const discoveryTab = document.querySelector('.activity-tab[data-activity="discovery"]');
        const liveFeedTab = document.querySelector('.explorer-tab-btn[data-tab="live-feed"]');
        const mementoInfoTab = document.querySelector('.live-feed-tab-btn[data-tab="memento-info"]');
        
        if (discoveryTab) {
          discoveryTab.click();
        }
        
        if (liveFeedTab) {
          liveFeedTab.click();
        }
        
        if (mementoInfoTab) {
          mementoInfoTab.click();
        }

        // Show live feed container
        const liveFeedContainer = document.querySelector('.live-feed-container');
        if (liveFeedContainer) {
          liveFeedContainer.classList.remove('hidden');
        }

        // Display memento details in live feed
        const mementoInfoContent = document.getElementById('memento-info-content');
        const mementoInfoList = mementoInfoContent?.querySelector('.memento-info-list');
        if (mementoInfoList) {
          displayMementoInLiveFeed(memento);
        }
      } catch (error) {
        console.error('Error handling map interaction:', error);
        showToast('Error displaying location on map. Please try again.', 'error');
      }
    });
  }
  
  return mementoElement;
}

// Show memento details
function showMementoDetails(memento) {
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
  
  // Display memento details
  if (mementoInfoList) {
    displayMementoInLiveFeed(memento);
  }
}

// Refresh recommendations
async function refreshRecommendations() {
  try {
    const recommendedList = document.querySelector('#recommended-content .curated-list');
    if (!recommendedList) return;
    
    // Show loading state
    recommendedList.innerHTML = `
      <div class="recommended-message">
        <i class="fas fa-spinner fa-spin"></i>
        <p>Refreshing recommendations...</p>
      </div>
    `;
    
    // Reload recommendations
    await loadRecommendedContent();
    
    // Show success message
    showToast('Recommendations refreshed', 'success');
  } catch (error) {
    console.error('Error refreshing recommendations:', error);
    showToast('Error refreshing recommendations', 'error');
  }
}

// Load near me content
async function loadNearMeContent() {
  try {
    const nearMeList = document.querySelector('#near-me-content .curated-list');
    if (!nearMeList) return;
    
    // Show loading state
    nearMeList.innerHTML = `
      <div class="near-me-message">
        <i class="fas fa-spinner fa-spin"></i>
        <p>Loading nearby mementos...</p>
      </div>
    `;
    
    // Get user's location
    const userLocation = await getUserLocation();
    if (!userLocation) {
      nearMeList.innerHTML = `
        <div class="near-me-message">
          <i class="fas fa-map-marker-alt"></i>
          <p>Please enable location services to see nearby mementos</p>
        </div>
      `;
      return;
    }
    
    // Get all mementos using the existing function
    const allMementos = await getAllMementos();
    
    // Clear existing content
    nearMeList.innerHTML = '';
    
    // Remove any existing pagination
    const existingPagination = nearMeList.parentElement.querySelector('.near-me-pagination');
    if (existingPagination) {
      existingPagination.remove();
    }
    
    // Calculate distance and sort mementos
    const mementos = allMementos
      .map(memento => {
        if (!memento.location || !memento.location.coordinates) return null;
        
        const distance = calculateDistance(
          userLocation,
          memento.location.coordinates
        );
        
        return {
          ...memento,
          distance
        };
      })
      .filter(memento => memento !== null)
      .sort((a, b) => a.distance - b.distance)
      .slice(0, 5); // Show top 5 nearest mementos
    
    // Create pagination container
    const paginationContainer = document.createElement('div');
    paginationContainer.className = 'near-me-pagination';
    
    // Create pagination buttons
    for (let i = 0; i < mementos.length; i++) {
      const button = document.createElement('button');
      button.className = 'pagination-btn';
      button.textContent = i + 1;
      button.title = `View nearby memento #${i + 1}`;
      
      // Add click handler
      button.addEventListener('click', () => {
        // Remove active class from all buttons
        paginationContainer.querySelectorAll('.pagination-btn').forEach(btn => {
          btn.classList.remove('active');
        });
        
        // Add active class to clicked button
        button.classList.add('active');
        
        // Scroll to the corresponding memento
        const mementoElement = nearMeList.children[i];
        if (mementoElement) {
          mementoElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
      
      paginationContainer.appendChild(button);
    }

    // Add refresh button to pagination container
    const refreshBtn = document.createElement('button');
    refreshBtn.className = 'refresh-btn';
    refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i>';
    refreshBtn.title = 'Refresh nearby mementos';
    refreshBtn.addEventListener('click', (e) => {
      e.preventDefault();
      refreshNearMeContent();
    });
    paginationContainer.appendChild(refreshBtn);
    
    // Add pagination container before the list
    nearMeList.parentElement.insertBefore(paginationContainer, nearMeList);
    
    // Add each memento to the list
    for (const memento of mementos) {
      const mementoElement = await createCuratedMementoElement(memento);
      nearMeList.appendChild(mementoElement);
    }

    // Setup scroll-based pagination
    setupScrollBasedPagination(nearMeList, paginationContainer);

    // If no mementos found, show a message but keep the pagination
    if (mementos.length === 0) {
      nearMeList.innerHTML = `
        <div class="near-me-message">
          <i class="fas fa-map-marker-alt"></i>
          <p>No mementos found nearby.<br>Be the first to create one!</p>
        </div>
      `;
    }
  } catch (error) {
    console.error('Error loading near me content:', error);
    showToast('Error loading nearby mementos', 'error');
  }
}

// Refresh near me content
async function refreshNearMeContent() {
  try {
    const nearMeList = document.querySelector('#near-me-content .curated-list');
    if (!nearMeList) return;
    
    // Show loading state
    nearMeList.innerHTML = `
      <div class="near-me-message">
        <i class="fas fa-spinner fa-spin"></i>
        <p>Refreshing nearby mementos...</p>
      </div>
    `;
    
    // Reload near me content
    await loadNearMeContent();
    
    // Show success message
    showToast('Nearby mementos refreshed', 'success');
  } catch (error) {
    console.error('Error refreshing near me content:', error);
    showToast('Error refreshing nearby mementos', 'error');
  }
}

// Get user's location
async function getUserLocation() {
  try {
    return new Promise((resolve, reject) => {
      if (!navigator.geolocation) {
        reject(new Error('Geolocation is not supported by your browser'));
        return;
      }

      navigator.geolocation.getCurrentPosition(
        (position) => {
          resolve({
            latitude: position.coords.latitude,
            longitude: position.coords.longitude
          });
        },
        (error) => {
          reject(error);
        }
      );
    });
  } catch (error) {
    console.error('Error getting user location:', error);
    return null;
  }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  // Check if user is authenticated
  firebase.auth().onAuthStateChanged((user) => {
    if (user) {
      // User is signed in, initialize curated
      initializeCurated();
    } else {
      // User is not signed in, show sign in message
      const curatedContainer = document.querySelector('.curated-container');
      if (curatedContainer) {
        curatedContainer.innerHTML = `
          <div class="curated-message">
            <i class="fas fa-user-lock"></i>
            <p>Please sign in to view curated mementos</p>
          </div>
        `;
      }
    }
  });
}); 