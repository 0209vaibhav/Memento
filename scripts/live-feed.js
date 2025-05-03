// Global variables
let currentRadius = null;
let mapboxMap = null;
let isRadiusEnabled = true; // Set to true by default

// Load categories, tags, and durations from JSON files
let allCategories = [];
let allTags = [];
let allDurations = [];

// Function to load categories, tags, and durations
async function loadCategoriesAndTags() {
  try {
    const categoriesResponse = await fetch('memento_categories.json');
    const tagsResponse = await fetch('memento_tags.json');
    const durationsResponse = await fetch('memento_durations.json');
    
    allCategories = await categoriesResponse.json();
    allTags = await tagsResponse.json();
    allDurations = await durationsResponse.json();
    
    // Make them available globally
    window.categories = allCategories;
    window.tags = allTags;
    window.durations = allDurations;
    
    console.log('Loaded categories:', allCategories);
    console.log('Loaded tags:', allTags);
    console.log('Loaded durations:', allDurations);
  } catch (error) {
    console.error('Error loading categories, tags, and durations:', error);
  }
}

// Call the function when the script loads
loadCategoriesAndTags();

// Live Feed Tabs Functionality
function initializeLiveFeedTabs() {
  const tabButtons = document.querySelectorAll('.live-feed-tab-btn');
  const tabContents = document.querySelectorAll('.live-feed-tab-content');

  // Set preview as the default active tab
  const previewButton = document.querySelector('.live-feed-tab-btn[data-tab="preview"]');
  const previewContent = document.getElementById('preview-content');
  if (previewButton && previewContent) {
    previewButton.classList.add('active');
    previewContent.classList.add('active');
    // Load preview content only once during initialization
    loadPreviewContent();
  }

  // Add total mementos header above search bar
  updateTotalMementosHeader();

  tabButtons.forEach(button => {
    button.addEventListener('click', (e) => {
      e.preventDefault(); // Prevent default behavior
      e.stopPropagation(); // Stop event propagation
      
      const tabId = button.getAttribute('data-tab');
      console.log('Tab clicked:', tabId);
      
      // Remove active class from all buttons and contents
      tabButtons.forEach(btn => btn.classList.remove('active'));
      tabContents.forEach(content => {
        content.classList.remove('active');
        content.style.display = 'none';
      });

      // Add active class to clicked button
      button.classList.add('active');

      // Show corresponding content
      const content = document.getElementById(`${tabId}-content`);
      if (content) {
        content.classList.add('active');
        content.style.display = 'block';
        console.log('Content shown:', `${tabId}-content`);
      }

      // Load content based on tab
      switch (tabId) {
        case 'categories':
          loadCategoriesContent();
          break;
        case 'tags':
          loadTagsContent();
          break;
        case 'duration':
          loadDurationContent();
          break;
        case 'preview':
          loadPreviewContent();
          break;
        case 'memento-info':
          // Ensure memento-info content is visible
          const mementoInfoContent = document.getElementById('memento-info-content');
          if (mementoInfoContent) {
            mementoInfoContent.classList.add('active');
            mementoInfoContent.style.display = 'block';
          }
          break;
      }
    });
  });
}

// Add event listener for DOMContentLoaded to ensure proper initialization
document.addEventListener('DOMContentLoaded', () => {
  console.log('Initializing live feed tabs...');
  initializeLiveFeedTabs();
});

// Function to update total mementos header
async function updateTotalMementosHeader() {
  const liveFeedContainer = document.querySelector('.live-feed-container');
  if (!liveFeedContainer) return;

  // Get all mementos within current radius
  const mementos = await getAllMementos();
  
  // Get creator filter states
  const showMyMementos = document.getElementById('filter-my-mementos')?.checked ?? true;
  const showAllUsers = document.getElementById('filter-all-users')?.checked ?? true;
  const showPublic = document.getElementById('filter-public')?.checked ?? true;
  
  // Get current map center and radius
  const mapElement = document.getElementById('map');
  const mapInstance = mapElement?._mapboxgl_map;
  const center = mapInstance?.getCenter();
  const radiusSlider = document.getElementById('radius-slider');
  const currentRadius = radiusSlider ? parseFloat(radiusSlider.value) : 0;
  
  // Filter mementos based on location and creator type
  const filteredMementos = mementos.filter(memento => {
    // Skip if no location data
    if (!memento.location || !memento.location.coordinates) return false;

    // Validate coordinates
    const { latitude, longitude } = memento.location.coordinates;
    if (typeof latitude !== 'number' || typeof longitude !== 'number' || 
        isNaN(latitude) || isNaN(longitude)) {
      return false;
    }

    // Check if within radius
    if (center && currentRadius) {
      try {
        const distance = turf.distance(
          turf.point([center.lng, center.lat]),
          turf.point([longitude, latitude]),
          { units: 'miles' }
        );
        if (distance > currentRadius) return false;
      } catch (error) {
        console.warn('Error calculating distance:', error);
        return false;
      }
    }

    // Check creator filters
    const isCurrentUserMemento = memento.userId === firebase.auth().currentUser?.uid;
    const isOtherUserMemento = memento.userId && memento.userId !== firebase.auth().currentUser?.uid;
    const isPublicMemento = memento.isPublic === true;

    // Include if it matches any enabled filter
    return (showMyMementos && isCurrentUserMemento) ||
           (showAllUsers && isOtherUserMemento) ||
           (showPublic && isPublicMemento);
  });
  
  // Format radius value
  const radiusText = `${currentRadius.toFixed(2)} mi`;
  
  // Create or update header
  let totalCountHeader = liveFeedContainer.querySelector('.total-mementos-header');
  if (!totalCountHeader) {
    totalCountHeader = document.createElement('div');
    totalCountHeader.className = 'total-mementos-header';
    // Insert before search bar
    const searchBar = liveFeedContainer.querySelector('.search-bar-container');
    if (searchBar) {
      liveFeedContainer.insertBefore(totalCountHeader, searchBar);
    }
  }
  
  totalCountHeader.innerHTML = `
    <div class="total-mementos-content">
      <span class="total-mementos-icon"><i class="fas fa-layer-group"></i></span>
      <span class="total-mementos-text">Mementos within ${radiusText}:</span>
      <span class="total-mementos-count">${filteredMementos.length}</span>
    </div>
  `;
}

// Load Categories Content
async function loadCategoriesContent() {
  // First ensure the categories content container exists
  const categoriesContent = document.getElementById('categories-content');
  if (!categoriesContent) {
    console.error('Categories content container not found');
    return;
  }
  
  // Create categories list if it doesn't exist
  let categoriesList = categoriesContent.querySelector('.categories-list');
  if (!categoriesList) {
    categoriesList = document.createElement('div');
    categoriesList.className = 'categories-list';
    categoriesContent.appendChild(categoriesList);
  }
  
  try {
  // Clear existing content
  categoriesList.innerHTML = '';

  // Get all mementos within current radius
  const mementos = await getAllMementos();
    
    // Get creator filter states
    const showMyMementos = document.getElementById('filter-my-mementos')?.checked ?? true;
    const showAllUsers = document.getElementById('filter-all-users')?.checked ?? true;
    const showPublic = document.getElementById('filter-public')?.checked ?? true;
    
    // Get current map center and radius
    const mapElement = document.getElementById('map');
    const mapInstance = mapElement?._mapboxgl_map;
    const center = mapInstance?.getCenter();
    const radiusSlider = document.getElementById('radius-slider');
    const currentRadius = radiusSlider ? parseFloat(radiusSlider.value) : 0;
    
    console.log('Map state:', {
      center: center ? [center.lng, center.lat] : null,
      radius: currentRadius,
      totalMementos: mementos.length
    });

    // Filter mementos based on location and creator type
    const filteredMementos = mementos.filter(memento => {
      // Skip if no location data
      if (!memento.location || !memento.location.coordinates) return false;

      // Validate coordinates
      const { latitude, longitude } = memento.location.coordinates;
      if (typeof latitude !== 'number' || typeof longitude !== 'number' || 
          isNaN(latitude) || isNaN(longitude)) {
        return false;
      }

      // Check if within radius
      if (center && currentRadius) {
        try {
          const distance = turf.distance(
            turf.point([center.lng, center.lat]),
            turf.point([longitude, latitude]),
            { units: 'miles' }
          );
          if (distance > currentRadius) return false;
        } catch (error) {
          console.warn('Error calculating distance:', error);
          return false;
        }
      }

      // Check creator filters
      const isCurrentUserMemento = memento.userId === firebase.auth().currentUser?.uid;
      const isOtherUserMemento = memento.userId && memento.userId !== firebase.auth().currentUser?.uid;
      const isPublicMemento = memento.isPublic === true;

      // Include if it matches any enabled filter
      return (showMyMementos && isCurrentUserMemento) ||
             (showAllUsers && isOtherUserMemento) ||
             (showPublic && isPublicMemento);
    });
    
    // If no mementos, show message and return
    if (filteredMementos.length === 0) {
      const noMementosMessage = document.createElement('div');
      noMementosMessage.className = 'no-mementos-message';
      noMementosMessage.innerHTML = `
        <i class="fas fa-folder icon"></i>
        <p>No mementos found within ${currentRadius.toFixed(2)} miles</p>
        <p class="suggestion">Try adjusting the radius or moving to a different location</p>
      `;
      categoriesList.appendChild(noMementosMessage);
      return;
    }
  
  // Initialize counts for all categories
  const categoryCounts = {};
  window.categories.forEach(category => {
    categoryCounts[category.id] = 0;
  });
  
  // Count mementos by category
    filteredMementos.forEach(memento => {
    if (memento.category && categoryCounts.hasOwnProperty(memento.category)) {
      categoryCounts[memento.category]++;
    }
  });

    // Convert to array and sort by count in descending order
    const sortedCategories = window.categories
      .map(category => ({
        ...category,
        count: categoryCounts[category.id]
      }))
      .sort((a, b) => b.count - a.count);

    // Create document fragment for better performance
    const fragment = document.createDocumentFragment();

    // Create category items
  sortedCategories.forEach(category => {
    const categoryItem = document.createElement('div');
    categoryItem.className = 'category-item';
    categoryItem.innerHTML = `
      <div class="category-content">
        <span class="category-symbol">${category.symbol}</span>
        <span class="category-name">${category.name}</span>
        <span class="category-count">${category.count}</span>
      </div>
    `;

    // Add click event listener to prevent event propagation
    categoryItem.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      // Add your category item click handling logic here if needed
    });

    fragment.appendChild(categoryItem);
  });

    // Append all items at once
    categoriesList.appendChild(fragment);

  } catch (error) {
    console.error('Error loading categories content:', error);
    categoriesList.innerHTML = `
      <div class="no-my-mementos-message">
        <p>Error loading categories. Please try again.</p>
      </div>
    `;
  }
}

// Load Tags Content
async function loadTagsContent() {
  // First ensure the tags content container exists
  const tagsContent = document.getElementById('tags-content');
  if (!tagsContent) {
    console.error('Tags content container not found');
    return;
  }
  
  // Create tags list if it doesn't exist
  let tagsList = tagsContent.querySelector('.tags-list');
  if (!tagsList) {
    tagsList = document.createElement('div');
    tagsList.className = 'tags-list';
    tagsContent.appendChild(tagsList);
  }
  
  try {
    // Clear existing content
  tagsList.innerHTML = '';

  // Get all mementos within current radius
  const mementos = await getAllMementos();
    
    // Get creator filter states
    const showMyMementos = document.getElementById('filter-my-mementos')?.checked ?? true;
    const showAllUsers = document.getElementById('filter-all-users')?.checked ?? true;
    const showPublic = document.getElementById('filter-public')?.checked ?? true;
    
    // Get current map center and radius
    const mapElement = document.getElementById('map');
    const mapInstance = mapElement?._mapboxgl_map;
    const center = mapInstance?.getCenter();
    const radiusSlider = document.getElementById('radius-slider');
    const currentRadius = radiusSlider ? parseFloat(radiusSlider.value) : 0;
    
    console.log('Map state:', {
      center: center ? [center.lng, center.lat] : null,
      radius: currentRadius,
      totalMementos: mementos.length
    });

    // Filter mementos based on location and creator type
    const filteredMementos = mementos.filter(memento => {
      // Skip if no location data
      if (!memento.location || !memento.location.coordinates) return false;

      // Validate coordinates
      const { latitude, longitude } = memento.location.coordinates;
      if (typeof latitude !== 'number' || typeof longitude !== 'number' || 
          isNaN(latitude) || isNaN(longitude)) {
        return false;
      }

      // Check if within radius
      if (center && currentRadius) {
        try {
          const distance = turf.distance(
            turf.point([center.lng, center.lat]),
            turf.point([longitude, latitude]),
            { units: 'miles' }
          );
          if (distance > currentRadius) return false;
        } catch (error) {
          console.warn('Error calculating distance:', error);
          return false;
        }
      }

      // Check creator filters
      const isCurrentUserMemento = memento.userId === firebase.auth().currentUser?.uid;
      const isOtherUserMemento = memento.userId && memento.userId !== firebase.auth().currentUser?.uid;
      const isPublicMemento = memento.isPublic === true;

      // Include if it matches any enabled filter
      return (showMyMementos && isCurrentUserMemento) ||
             (showAllUsers && isOtherUserMemento) ||
             (showPublic && isPublicMemento);
    });
    
    // If no mementos, show message and return
    if (filteredMementos.length === 0) {
      const noMementosMessage = document.createElement('div');
      noMementosMessage.className = 'no-mementos-message';
      noMementosMessage.innerHTML = `
        <i class="fas fa-tags icon"></i>
        <p>No mementos found within ${currentRadius.toFixed(2)} miles</p>
        <p class="suggestion">Try adjusting the radius or<br>moving to a different location</p>
      `;
      tagsList.appendChild(noMementosMessage);
      return;
    }
  
  // Initialize counts for all tags
  const tagCounts = {};
  window.tags.forEach(tag => {
    tagCounts[tag.id] = 0;
  });
  
  // Count mementos by tag
    filteredMementos.forEach(memento => {
    if (memento.tags) {
      memento.tags.forEach(tag => {
        if (tagCounts.hasOwnProperty(tag)) {
          tagCounts[tag]++;
        }
      });
    }
  });

  // Convert to array and sort by count in descending order
  const sortedTags = window.tags
    .map(tag => ({
      ...tag,
      count: tagCounts[tag.id]
    }))
    .sort((a, b) => b.count - a.count);

    // Create document fragment for better performance
    const fragment = document.createDocumentFragment();

  // Create tag items
  sortedTags.forEach(tag => {
    const tagItem = document.createElement('div');
    tagItem.className = 'tag-item';
    tagItem.innerHTML = `
      <div class="tag-content">
        <span class="tag-symbol">${tag.symbol}</span>
        <span class="tag-name">${tag.name}</span>
        <span class="tag-count">${tag.count}</span>
      </div>
    `;

    // Add click event listener to prevent event propagation
    tagItem.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      // Add your tag item click handling logic here if needed
    });

    fragment.appendChild(tagItem);
  });

    // Append all items at once
    tagsList.appendChild(fragment);

  } catch (error) {
    console.error('Error loading tags content:', error);
    tagsList.innerHTML = `
      <div class="no-my-mementos-message">
        <p>Error loading tags. Please try again.</p>
      </div>
    `;
  }
}

// Load Duration Content
async function loadDurationContent() {
  // First ensure the duration content container exists
  const durationContent = document.getElementById('duration-content');
  if (!durationContent) {
    console.error('Duration content container not found');
    return;
  }
  
  // Create duration list if it doesn't exist
  let durationList = durationContent.querySelector('.duration-list');
  if (!durationList) {
    durationList = document.createElement('div');
    durationList.className = 'duration-list';
    durationContent.appendChild(durationList);
  }
  
  try {
    // Clear existing content
  durationList.innerHTML = '';

  // Get all mementos within current radius
  const mementos = await getAllMementos();
    
    // Get creator filter states
    const showMyMementos = document.getElementById('filter-my-mementos')?.checked ?? true;
    const showAllUsers = document.getElementById('filter-all-users')?.checked ?? true;
    const showPublic = document.getElementById('filter-public')?.checked ?? true;
    
    // Get current map center and radius
    const mapElement = document.getElementById('map');
    const mapInstance = mapElement?._mapboxgl_map;
    const center = mapInstance?.getCenter();
    const radiusSlider = document.getElementById('radius-slider');
    const currentRadius = radiusSlider ? parseFloat(radiusSlider.value) : 0;
    
    console.log('Map state:', {
      center: center ? [center.lng, center.lat] : null,
      radius: currentRadius,
      totalMementos: mementos.length
    });

    // Filter mementos based on location and creator type
    const filteredMementos = mementos.filter(memento => {
      // Skip if no location data
      if (!memento.location || !memento.location.coordinates) return false;

      // Validate coordinates
      const { latitude, longitude } = memento.location.coordinates;
      if (typeof latitude !== 'number' || typeof longitude !== 'number' || 
          isNaN(latitude) || isNaN(longitude)) {
        return false;
      }

      // Check if within radius
      if (center && currentRadius) {
        try {
          const distance = turf.distance(
            turf.point([center.lng, center.lat]),
            turf.point([longitude, latitude]),
            { units: 'miles' }
          );
          if (distance > currentRadius) return false;
        } catch (error) {
          console.warn('Error calculating distance:', error);
          return false;
        }
      }

      // Check creator filters
      const isCurrentUserMemento = memento.userId === firebase.auth().currentUser?.uid;
      const isOtherUserMemento = memento.userId && memento.userId !== firebase.auth().currentUser?.uid;
      const isPublicMemento = memento.isPublic === true;

      // Include if it matches any enabled filter
      return (showMyMementos && isCurrentUserMemento) ||
             (showAllUsers && isOtherUserMemento) ||
             (showPublic && isPublicMemento);
    });
    
    // If no mementos, show message and return
    if (filteredMementos.length === 0) {
      const noMementosMessage = document.createElement('div');
      noMementosMessage.className = 'no-mementos-message';
      noMementosMessage.innerHTML = `
        <i class="fas fa-hourglass-half icon"></i>
        <p>No mementos found within ${currentRadius.toFixed(2)} miles</p>
        <p class="suggestion">Try adjusting the radius or<br>moving to a different location</p>
      `;
      durationList.appendChild(noMementosMessage);
      return;
    }
  
  // Initialize counts for all durations
  const durationCounts = {};
  allDurations.forEach(duration => {
    durationCounts[duration.id] = 0;
  });
  
  // Count mementos by duration
    filteredMementos.forEach(memento => {
    if (memento.duration && durationCounts.hasOwnProperty(memento.duration)) {
      durationCounts[memento.duration]++;
    }
  });

  // Convert to array and sort by count in descending order
  const sortedDurations = allDurations
    .map(duration => ({
      ...duration,
      count: durationCounts[duration.id]
    }))
    .sort((a, b) => b.count - a.count);

    // Create document fragment for better performance
    const fragment = document.createDocumentFragment();

  // Create duration items
  sortedDurations.forEach(duration => {
    const durationItem = document.createElement('div');
    durationItem.className = 'duration-item';
    durationItem.innerHTML = `
      <div class="duration-content">
        <span class="duration-symbol">${duration.symbol}</span>
        <span class="duration-name">${duration.name}</span>
        <span class="duration-count">${duration.count}</span>
      </div>
    `;

    // Add click event listener to prevent event propagation
    durationItem.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      // Add your duration item click handling logic here if needed
    });

    fragment.appendChild(durationItem);
  });

    // Append all items at once
    durationList.appendChild(fragment);

  } catch (error) {
    console.error('Error loading duration content:', error);
    durationList.innerHTML = `
      <div class="no-my-mementos-message">
        <p>Error loading duration content. Please try again.</p>
      </div>
    `;
  }
}

// Load Preview Content
async function loadPreviewContent() {
  // First ensure the preview content container exists
  const previewContent = document.getElementById('preview-content');
  if (!previewContent) {
    console.error('Preview content container not found');
    return;
  }

  // Create preview list if it doesn't exist
  let previewList = previewContent.querySelector('.preview-list');
  if (!previewList) {
    previewList = document.createElement('div');
    previewList.className = 'preview-list';
    previewContent.appendChild(previewList);
  }
  
  try {
    // Clear existing content
  previewList.innerHTML = '';

  // Get all mementos within current radius
  const mementos = await getAllMementos();
    
    // Get creator filter states
    const showMyMementos = document.getElementById('filter-my-mementos')?.checked ?? true;
    const showAllUsers = document.getElementById('filter-all-users')?.checked ?? true;
    const showPublic = document.getElementById('filter-public')?.checked ?? true;
    
    // Get current map center and radius
    const mapElement = document.getElementById('map');
    const mapInstance = mapElement?._mapboxgl_map;
    const center = mapInstance?.getCenter();
    const radiusSlider = document.getElementById('radius-slider');
    const currentRadius = radiusSlider ? parseFloat(radiusSlider.value) : 0;
    
    console.log('Map state:', {
      center: center ? [center.lng, center.lat] : null,
      radius: currentRadius,
      totalMementos: mementos.length
    });

    // Filter mementos based on location and creator type
    const filteredMementos = mementos.filter(memento => {
      // Skip if no location data
      if (!memento.location || !memento.location.coordinates) return false;

      // Validate coordinates
      const { latitude, longitude } = memento.location.coordinates;
      if (typeof latitude !== 'number' || typeof longitude !== 'number' || 
          isNaN(latitude) || isNaN(longitude)) {
        return false;
      }

      // Check if within radius
      if (center && currentRadius) {
        try {
          const distance = turf.distance(
            turf.point([center.lng, center.lat]),
            turf.point([longitude, latitude]),
            { units: 'miles' }
          );
          if (distance > currentRadius) return false;
        } catch (error) {
          console.warn('Error calculating distance:', error);
          return false;
        }
      }

      // Check creator filters
      const isCurrentUserMemento = memento.userId === firebase.auth().currentUser?.uid;
      const isOtherUserMemento = memento.userId && memento.userId !== firebase.auth().currentUser?.uid;
      const isPublicMemento = memento.isPublic === true;

      // Include if it matches any enabled filter
      return (showMyMementos && isCurrentUserMemento) ||
             (showAllUsers && isOtherUserMemento) ||
             (showPublic && isPublicMemento);
    });
  
  // Sort mementos by timestamp (most recent first)
    const sortedMementos = filteredMementos.sort((a, b) => {
    return new Date(b.timestamp) - new Date(a.timestamp);
  });

  // Create preview items
    for (const memento of sortedMementos) {
    const previewItem = document.createElement('div');
    previewItem.className = 'preview-item';
    
    // Format timestamp
    const timestamp = new Date(memento.timestamp);
    const timeAgo = getTimeAgo(timestamp);

      // Get username if available
      let username = 'Anonymous';
      
      // Log the memento object to debug
      console.log('Memento data:', {
        id: memento.id,
        type: memento.mementoType,
        userId: memento.userId,
        username: memento.username,
        data: memento
      });

      if (memento.mementoType === 'public') {
        // For public mementos from JSON
        if (memento.username) {
          username = memento.username;
        } else if (memento.creator && memento.creator.username) {
          username = memento.creator.username;
        }
      } else if (memento.userId) {
        // For user-mementos and all-user-mementos from Firebase
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

      // Handle media content
      let mediaHtml = '<div class="placeholder-media"><i class="fas fa-eye"></i></div>';
      if (memento.media && Array.isArray(memento.media) && memento.media.length > 0) {
        const firstMedia = memento.media[0];
        if (typeof firstMedia === 'string') {
          // If media is a direct URL string
          mediaHtml = `<img src="${firstMedia}" alt="${memento.name || 'Memento'}" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\\'placeholder-media\\'><i class=\\'fas fa-eye\\'></i></div>';">`;
        } else if (firstMedia && typeof firstMedia === 'object') {
          // If media is an object with url property
          const mediaUrl = firstMedia.url || firstMedia.path || firstMedia.src;
          if (mediaUrl) {
            if (firstMedia.type && firstMedia.type.startsWith('video/')) {
              mediaHtml = `<video src="${mediaUrl}" controls></video>`;
            } else {
              mediaHtml = `<img src="${mediaUrl}" alt="${memento.name || 'Memento'}" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\\'placeholder-media\\'><i class=\\'fas fa-eye\\'></i></div>';">`;
            }
          }
        }
      }

      // Get category and tag symbols
      let categorySymbol = '';
      let tagSymbols = [];

      if (memento.mementoType === 'public') {
        // For public mementos, use the category and tag strings directly
        if (memento.category) {
          categorySymbol = memento.category.split(' ')[0]; // Get the emoji part
        }
        if (memento.mementoTags && memento.mementoTags.length > 0) {
          tagSymbols = memento.mementoTags.map(tag => tag.split(' ')[0]); // Get the emoji part
        }
      } else {
        // For user mementos, use the lookup
        if (memento.category) {
          const categoryInfo = allCategories.find(c => c.id === memento.category);
          categorySymbol = categoryInfo ? categoryInfo.symbol : '';
        }
        if (memento.tags && memento.tags.length > 0) {
          tagSymbols = memento.tags.map(tag => {
            const tagInfo = allTags.find(t => t.id === tag);
            return tagInfo ? tagInfo.symbol : '';
          });
        }
      }

      const mementoHtml = `
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
            </div>
          </div>
          <div class="memento-content">
            <div class="memento-details">
              <h3 class="memento-name">
                <span>${memento.name || 'Untitled Memento'}</span>
              </h3>
              <div class="memento-attributes">
                ${categorySymbol ? `
                  <span class="memento-category">
                    ${categorySymbol}
                  </span>
                ` : ''}
                ${tagSymbols.length > 0 ? `
                  <span class="memento-tags">
                    ${tagSymbols.map(symbol => `<span class="tag">${symbol}</span>`).join('')}
                  </span>
                ` : ''}
              </div>
            </div>
          </div>
        </div>
      `;

      previewItem.innerHTML = mementoHtml;

      // Add click event listener to the entire preview item
      previewItem.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();

        // Use the unified display function
        displayMementoInfo(memento);
      });

      // Add event listeners for the view on map button
      if (memento.location && memento.location.coordinates) {
        const viewOnMapBtn = previewItem.querySelector('.view-on-map-btn');
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
            } catch (error) {
              console.error('Error handling map interaction:', error);
              showToast('Error displaying location on map. Please try again.', 'error');
            }
          });
        }
      }

      previewList.appendChild(previewItem);
    }

    // Show message if no mementos in radius
    if (sortedMementos.length === 0) {
      const noMementosMessage = document.createElement('div');
      noMementosMessage.className = 'no-mementos-message';
      noMementosMessage.innerHTML = `
        <i class="fas fa-eye icon"></i>
        <p>No mementos found within ${currentRadius.toFixed(2)} miles</p>
        <p class="suggestion">Try adjusting the radius or<br>moving to a different location</p>
      `;
      previewList.appendChild(noMementosMessage);
    }

  } catch (error) {
    console.error('Error loading preview content:', error);
    const noMementosMessage = document.createElement('div');
    noMementosMessage.className = 'no-mementos-message';
    noMementosMessage.innerHTML = `
      <i class="fas fa-exclamation-circle icon"></i>
      <p>Error loading preview content</p>
      <p class="suggestion">Please try again</p>
    `;
    previewList.appendChild(noMementosMessage);
  }
}

// Helper function to format time ago
function getTimeAgo(timestamp) {
  const now = new Date();
  const diffInSeconds = Math.floor((now - timestamp) / 1000);
  
  if (diffInSeconds < 60) return 'Just now';
  if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`;
  if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`;
  if (diffInSeconds < 604800) return `${Math.floor(diffInSeconds / 86400)}d ago`;
  if (diffInSeconds < 2592000) return `${Math.floor(diffInSeconds / 604800)}w ago`;
  if (diffInSeconds < 31536000) return `${Math.floor(diffInSeconds / 2592000)}mo ago`;
  return `${Math.floor(diffInSeconds / 31536000)}y ago`;
}

// Get all mementos from Firebase within current radius
async function getAllMementos() {
  try {
    // Get creator filter states
    const showMyMementos = document.getElementById('filter-my-mementos')?.checked ?? true;
    const showAllUsers = document.getElementById('filter-all-users')?.checked ?? true;
    const showPublic = document.getElementById('filter-public')?.checked ?? true;
    
    // Get radius filter state
    const radiusToggle = document.getElementById('radius-toggle');
    isRadiusEnabled = radiusToggle ? radiusToggle.checked : true;
    
    let allMementos = [];
    const currentUser = firebase.auth().currentUser;
    
    // 1. Get user mementos (red squares) from Firebase if enabled
    if (showMyMementos && currentUser) {
      const userMementosSnapshot = await firebase.firestore()
        .collection('mementos')
        .where('userId', '==', currentUser.uid)
        .get();
      
      const userMementos = userMementosSnapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data(),
        mementoType: 'user'
      }));
      allMementos.push(...userMementos);
    }
    
    // 2. Get all-user mementos (blue rounded squares) from Firebase if enabled
    if (showAllUsers && currentUser) {
      const allUserMementosSnapshot = await firebase.firestore()
        .collection('mementos')
        .where('userId', '!=', currentUser.uid)
        .get();
      
      const allUserMementos = allUserMementosSnapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data(),
        mementoType: 'all-user'
      }));
      allMementos.push(...allUserMementos);
    }
    
    // 3. Get public mementos (green circles) from JSON files if enabled
    if (showPublic) {
      try {
        const publicMementos = await loadPublicMementosData();
        const formattedPublicMementos = publicMementos.map(event => ({
          ...event,
          mementoType: 'public',
          location: {
            coordinates: {
              latitude: event.location.latitude,
              longitude: event.location.longitude
            }
          }
        }));
        allMementos.push(...formattedPublicMementos);
      } catch (error) {
        console.error('Error loading public mementos:', error);
      }
    }
    
    // If radius is not enabled, return all filtered mementos
    if (!isRadiusEnabled || !currentRadius || !mapboxMap) {
      console.log('Radius filtering disabled, returning filtered mementos:', allMementos.length);
      return allMementos;
    }
    
    // Get current map center
    const center = mapboxMap.getCenter();
    const currentLocation = [center.lng, center.lat];
    
    console.log('Filtering by radius:', {
      currentLocation,
      currentRadius,
      totalMementos: allMementos.length
    });
    
    // Filter by radius
    const filteredMementos = allMementos.filter(memento => {
      if (!memento.location || !memento.location.coordinates) return false;
      
      // Validate coordinates
      const { latitude, longitude } = memento.location.coordinates;
      if (typeof latitude !== 'number' || typeof longitude !== 'number' || 
          isNaN(latitude) || isNaN(longitude)) {
        return false;
      }

      // Convert memento location to [lng, lat] format
      const mementoPoint = [
        longitude,
        latitude
      ];
      
      // Calculate distance using turf.js
      const distance = turf.distance(
        turf.point(currentLocation),
        turf.point(mementoPoint),
        { units: 'miles' }
      );
      
      const isWithinRadius = distance <= currentRadius;
      console.log('Distance check:', {
        name: memento.name,
        type: memento.mementoType,
        distance,
        radius: currentRadius,
        isWithinRadius
      });
      
      return isWithinRadius;
    });
    
    console.log('Final filtered mementos:', {
      total: filteredMementos.length,
      byType: {
        user: filteredMementos.filter(m => m.mementoType === 'user').length,
        allUser: filteredMementos.filter(m => m.mementoType === 'all-user').length,
        public: filteredMementos.filter(m => m.mementoType === 'public').length
      }
    });
    
    return filteredMementos;
  } catch (error) {
    console.error('Error fetching mementos:', error);
    return [];
  }
}

// Update refreshLiveFeedContent to use the existing updateRadiusCircle
function refreshLiveFeedContent() {
  const activeTab = document.querySelector('.live-feed-tab-btn.active');
  if (!activeTab) {
    console.log('No active tab found');
    return;
  }
  
  const tabId = activeTab.getAttribute('data-tab');
  console.log('Refreshing live feed content for tab:', tabId);
  
  // Clear existing content before refreshing
  const contentElement = document.getElementById(`${tabId}-content`);
  if (!contentElement) {
    console.error(`${tabId}-content element not found`);
    return;
  }
  
  // Clear the content element
  contentElement.innerHTML = '';
  
  // Update total mementos header
  updateTotalMementosHeader();
  
  // Then refresh the active tab
  switch (tabId) {
    case 'categories':
      loadCategoriesContent();
      break;
    case 'tags':
      loadTagsContent();
      break;
    case 'duration':
      loadDurationContent();
      break;
    case 'preview':
      loadPreviewContent();
      break;
  }
}

// Export functions to make them accessible from script.js
window.initializeLiveFeedTabs = initializeLiveFeedTabs;
window.loadCategoriesContent = loadCategoriesContent;
window.loadTagsContent = loadTagsContent;
window.loadDurationContent = loadDurationContent;

// Make the function available globally
window.displayMementoInLiveFeed = function(memento) {
  // ... existing function code ...
}