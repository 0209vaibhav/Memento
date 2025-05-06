// Global marker arrays
let publicMementoMarkers = [];
let userMementoMarkers = [];
let allUserMementoMarkers = [];

// Global variables
let map;
let userLocation = null;
let liveLocationMarker = null;
let currentTab = 'explorer'; // Set default tab
let currentOpenPopup = null;
let favorites = JSON.parse(localStorage.getItem('favorites')) || [];
let heatmapLayer = null;
let isHeatmapVisible = false;

// Load categories and tags from JSON files
// let allCategories = [];

// Function to load categories and tags
async function loadCategoriesAndTags() {
  try {
    const categoriesResponse = await fetch('memento_categories.json');
    const tagsResponse = await fetch('memento_tags.json');
    
    if (!categoriesResponse.ok || !tagsResponse.ok) {
      throw new Error('Failed to load categories or tags');
    }
    
    window.categories = await categoriesResponse.json();
    window.tags = await tagsResponse.json();
    
    console.log('Loaded categories:', window.categories);
    console.log('Loaded tags:', window.tags);
  } catch (error) {
    console.error('Error loading categories and tags:', error);
    // Set default values to prevent undefined errors
    window.categories = [];
    window.tags = [];
  }
}

// Call the function when the script loads
loadCategoriesAndTags();

// Define all possible tags with their symbols if not already defined
if (typeof allTags === 'undefined') {
  const allTags = [
    { id: 'ephemeral', name: 'Ephemeral', symbol: '🌀' },
    { id: 'unmapped', name: 'Unmapped', symbol: '📍' },
    { id: 'niche', name: 'Niche/Cult', symbol: '🧬' },
    { id: 'emotional', name: 'Emotionally Charged', symbol: '💫' },
    { id: 'hidden', name: 'Hidden Gem', symbol: '🕵️' },
    { id: 'unexpected', name: 'Unexpected Encounter', symbol: '🎭' },
    { id: 'reflective', name: 'Reflective', symbol: '🪞' },
    { id: 'unpleasant', name: 'Unpleasant Truth', symbol: '💔' },
    { id: 'rare', name: 'Once-in-a-While', symbol: '⏳' }
  ];
}

// Function to update markers based on creator filters
function updateMarkersByCreator(showMyMementos, showAllUsers, showPublic) {
  console.log('Updating markers with settings:', { showMyMementos, showAllUsers, showPublic });
  
  if (!window.markers || !window.markers.length) {
    console.log('No markers to update');
    return;
  }

  console.log(`Total markers to process: ${window.markers.length}`);
  
  // Get current user ID for filtering
  const currentUserId = firebase.auth().currentUser?.uid;
  
  // Update all markers based on their type and filter settings
  window.markers.forEach(marker => {
    if (!marker || !marker.getElement) {
      console.log('Skipping invalid marker');
      return;
    }
    
    const markerElement = marker.getElement();
    if (!markerElement) {
      console.log('Skipping marker with no element');
      return;
    }
    
    // Get marker type from data attribute
    const markerType = markerElement.getAttribute('data-marker-type');
    console.log('Processing marker:', { markerType });
    
    // Determine if marker should be shown based on type and filter settings
    let shouldShow = false;
    
    if (markerType === 'user-memento') {
      shouldShow = showMyMementos;
      console.log('User memento - should show:', shouldShow);
    } else if (markerType === 'all-user-memento') {
      shouldShow = showAllUsers;
      console.log('All user memento - should show:', shouldShow);
    } else if (markerType === 'public-memento') {
      shouldShow = showPublic;
      console.log('Public memento - should show:', shouldShow);
    } else {
      console.log('Unknown marker type:', markerType);
    }
    
    // Update marker visibility
    markerElement.style.display = shouldShow ? 'block' : 'none';
    console.log('Updated marker visibility:', { markerType, shouldShow });
  });
}

// Add event listeners for creator filter toggles
function initializeCreatorFilterListeners() {
  const creatorsToggle = document.getElementById('creators-toggle');
  const myMementosToggle = document.getElementById('filter-my-mementos');
  const allUsersToggle = document.getElementById('filter-all-users');
  const publicToggle = document.getElementById('filter-public');
  
  // Handle main creators toggle
  if (creatorsToggle) {
    creatorsToggle.addEventListener('change', function() {
      const isEnabled = this.checked;
      
      // Update all markers based on individual toggle states
      const showMyMementos = isEnabled && myMementosToggle.checked;
      const showAllUsers = isEnabled && allUsersToggle.checked;
      const showPublic = isEnabled && publicToggle.checked;
      updateMarkersByCreator(showMyMementos, showAllUsers, showPublic);
    });
  }
  
  // Handle my mementos toggle
  if (myMementosToggle) {
    myMementosToggle.addEventListener('change', function() {
      const isEnabled = creatorsToggle.checked;
      if (isEnabled) {
        const showMyMementos = this.checked;
        const showAllUsers = allUsersToggle.checked;
        const showPublic = publicToggle.checked;
        updateMarkersByCreator(showMyMementos, showAllUsers, showPublic);
      }
    });
  }
  
  // Handle all users toggle
  if (allUsersToggle) {
    allUsersToggle.addEventListener('change', function() {
      const isEnabled = creatorsToggle.checked;
      if (isEnabled) {
        const showMyMementos = myMementosToggle.checked;
        const showAllUsers = this.checked;
        const showPublic = publicToggle.checked;
        updateMarkersByCreator(showMyMementos, showAllUsers, showPublic);
      }
    });
  }
  
  // Handle public toggle
  if (publicToggle) {
    publicToggle.addEventListener('change', function() {
      const isEnabled = creatorsToggle.checked;
      if (isEnabled) {
        const showMyMementos = myMementosToggle.checked;
        const showAllUsers = allUsersToggle.checked;
        const showPublic = this.checked;
        updateMarkersByCreator(showMyMementos, showAllUsers, showPublic);
      }
    });
  }
  
  // Initial update
  const showMyMementos = creatorsToggle.checked && myMementosToggle.checked;
  const showAllUsers = creatorsToggle.checked && allUsersToggle.checked;
  const showPublic = creatorsToggle.checked && publicToggle.checked;
  updateMarkersByCreator(showMyMementos, showAllUsers, showPublic);
}

// Handle splash screen
document.addEventListener('DOMContentLoaded', function() {
    const splashScreen = document.getElementById('splash-screen');
    
    if (splashScreen) {
        // Hide splash screen after animation completes
        setTimeout(function() {
            splashScreen.classList.add('splash-hidden');
            
            // Initialize creator filter listeners
            initializeCreatorFilterListeners();
        }, 4000); // Increased to 4 seconds
    } else {
        // If no splash screen, just initialize the filters
        initializeCreatorFilterListeners();
    }
});

document.addEventListener("DOMContentLoaded", function () {
  // Add link to memento markers styles
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = 'styles/memento-markers.css';
  document.head.appendChild(link);

  // Initialize journey capture functionality
  initializeJourneyCapture();

  // ---------------------------
  // 1) Global variables & state
  // ---------------------------
  // Variables to track current popups
  // (removed popup variables)

  // ---------------------------
  // 2) Grab all DOM references
  // ---------------------------
  const tabs = document.querySelectorAll(".activity-tab");
  const mainContent = document.querySelector(".main-content");
  const infoTab = document.querySelector(".info-tab");

  // ---------------------------
  // 3) Panel expansion function
  // ---------------------------
  function handlePanelControl(button, action) {
    button.addEventListener("touchstart", function(e) {
      e.preventDefault();
      e.stopPropagation();
      console.log(`${action} button touched`);
      
      const infoTab = document.querySelector('.info-tab');
      const expandBtn = document.getElementById('expand-left');
      const collapseBtn = document.getElementById('collapse-left');
      
      if (infoTab && expandBtn && collapseBtn) {
        if (action === 'expand') {
          infoTab.style.visibility = 'visible';
          infoTab.classList.remove('hidden');
          expandBtn.classList.add('hidden');
          collapseBtn.classList.remove('hidden');
        } else {
          infoTab.style.visibility = 'hidden';
          infoTab.classList.add('hidden');
          collapseBtn.classList.add('hidden');
          expandBtn.classList.remove('hidden');
        }
        
        if (map) {
          setTimeout(() => { map.resize(); }, 300);
        }
      }
    }, { passive: false });
    
    button.addEventListener("click", function(e) {
      e.preventDefault();
      e.stopPropagation();
      console.log(`${action} button clicked`);
      
      const infoTab = document.querySelector('.info-tab');
      const expandBtn = document.getElementById('expand-left');
      const collapseBtn = document.getElementById('collapse-left');
      
      if (infoTab && expandBtn && collapseBtn) {
        if (action === 'expand') {
          infoTab.style.visibility = 'visible';
          infoTab.classList.remove('hidden');
          expandBtn.classList.add('hidden');
          collapseBtn.classList.remove('hidden');
        } else {
          infoTab.style.visibility = 'hidden';
          infoTab.classList.add('hidden');
          collapseBtn.classList.add('hidden');
          expandBtn.classList.remove('hidden');
        }
        
        if (map) {
          setTimeout(() => { map.resize(); }, 300);
        }
      }
    });
  }

  // Initialize panel controls
  const panelExpandBtn = document.getElementById('expand-left');
  const panelCollapseBtn = document.getElementById('collapse-left');

  if (panelExpandBtn) {
    handlePanelControl(panelExpandBtn, 'expand');
  }

  if (panelCollapseBtn) {
    handlePanelControl(panelCollapseBtn, 'collapse');
  }

  // ---------------------------
  // 4) Firebase auth state listener
  // ---------------------------
  firebase.auth().onAuthStateChanged(async (user) => {
    // No settings loading needed
  });

  // Set initial state - Discovery tab and subtabs visible
  const discoveryTabElement = document.querySelector('.activity-tab[data-activity="discovery"]');
  if (discoveryTabElement) {
    // Activate Discovery tab
    tabs.forEach(tab => tab.classList.remove('active'));
    discoveryTabElement.classList.add('active');
    
    // Show Discovery subtabs
    document.querySelectorAll('.explorer-subtabs').forEach(subtab => {
      subtab.classList.remove('show');
    });
    const discoverySubtabs = discoveryTabElement.nextElementSibling;
    if (discoverySubtabs) {
      discoverySubtabs.classList.remove('hidden');
      discoverySubtabs.classList.add('show');
    }

    // Show filter settings container by default
    const filterSettingsContainer = document.querySelector('.filter-settings-container');
    if (filterSettingsContainer) {
      filterSettingsContainer.classList.remove('hidden');
      // Initialize filters
      initializeFilters();
    }
  }

  // Panel collapse/expand buttons
  const collapseLeftBtn = document.getElementById("collapse-left");
  const expandLeftBtn = document.getElementById("expand-left");
  const collapseRightBtn = document.getElementById("collapse-right");
  const expandRightBtn = document.getElementById("expand-right");

  // Radius controls
  const radiusControl = document.getElementById("radius-control");
  const radiusToggle = document.getElementById("radius-toggle");
  const radiusSlider = document.getElementById("radius-slider");
  const radiusValue = document.getElementById("radius-value");
  const radiusRangeToggle = document.getElementById('radius-range-toggle');
  const minValueSpan = document.querySelector('.radius-value-container .min-value');
  const maxValueSpan = document.querySelector('.radius-value-container .max-value');

  // Initialize default settings
  let savedSettings = {
      mapStyle: 'mapbox://styles/mapbox/light-v11',
      markerSize: 'medium',
      markerType: 'circle',
      enable3DBuildings: false,
      autoClosePopups: false,
      displayMode: 'expanded',
      showTimestamps: true
  };

  // ---------------------------
  // 5) Utility functions for unit conversion
  // ---------------------------
  // Function to convert feet to miles
  function feetToMiles(feet) {
      return feet / 5280;
  }

  // Function to convert miles to feet
  function milesToFeet(miles) {
      return miles * 5280;
  }

  // Function to update slider range based on selected range type
  function updateSliderRange(rangeType) {
    const radiusSlider = document.getElementById('radius-slider');
    const radiusValue = document.getElementById('radius-value');
    const minValueSpan = document.querySelector('.min-value');
    const maxValueSpan = document.querySelector('.max-value');
    
    if (!radiusSlider || !radiusValue || !minValueSpan || !maxValueSpan) return;
    
    if (rangeType === 'short') {
      // Short range: 10ft to 1mi
      radiusSlider.min = feetToMiles(10).toString();  // 10 feet in miles
      radiusSlider.max = "1";  // 1 mile
      radiusSlider.step = ((1 - feetToMiles(10)) / 100).toString();  // 100 steps
      radiusSlider.value = "0.5";  // Set default to middle of range
      minValueSpan.textContent = "10 ft";
      maxValueSpan.textContent = "1 mi";
      radiusValue.textContent = "0.5 mi";
    } else {
      // Long range: 1mi to 5mi
      radiusSlider.min = "1";
      radiusSlider.max = "5";
      radiusSlider.step = "0.1";  // 40 steps
      radiusSlider.value = "3";  // Set default to middle of range
      minValueSpan.textContent = "1 mi";
      maxValueSpan.textContent = "5 mi";
      radiusValue.textContent = "3 mi";
    }
    
    // Update the map with new radius
    if (map && map.getCenter()) {
      updateRadiusCircle();
    }
  }

  // Add event listener for range toggle
  document.getElementById('radius-range-toggle')?.addEventListener('change', function() {
    updateSliderRange(this.value);
  });

  // Add debounce function at the top level
  function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  // Optimize the radius slider event listener
  if (radiusSlider && radiusValue) {
    const debouncedUpdate = debounce((value) => {
      if (radiusToggle && radiusToggle.checked) {
          if (userLocation) {
          updateRadiusCircle(userLocation, parseFloat(value));
          } else if (map) {
              const center = map.getCenter().toArray();
          updateRadiusCircle(center, parseFloat(value));
        }
          }
          updateMap();
    }, 100); // 100ms debounce

    radiusSlider.addEventListener("input", function () {
      const formattedValue = Number(this.value).toFixed(2);
      radiusValue.innerText = formattedValue + ' mi';
      debouncedUpdate(formattedValue);
      });
  }

  // ---------------------------
  // 6) LocateMeControl (custom)
  // ---------------------------
  class LocateMeControl {
    onAdd(map) {
      this.map = map;
      this.container = document.createElement('div');
      this.container.className = 'mapboxgl-ctrl locate-me-control';

      // Create the button
      this.button = document.createElement('button');
      this.button.type = 'button';
      this.button.className = 'locate-me-button';
      // Icon + tooltip
      this.button.innerHTML = '<i class="fas fa-crosshairs"></i>';
      this.button.title = 'Your Location\nLearn more';

      // When clicked, fly to user location
      this.button.addEventListener('click', () => {
        if (userLocation) {
          // Ensure userLocation is in the correct format [longitude, latitude]
          const center = Array.isArray(userLocation) ? userLocation : [userLocation.lng, userLocation.lat];
          this.map.flyTo({ center, zoom: 14 });
        }
      });

      this.container.appendChild(this.button);
      return this.container;
    }
    onRemove() {
      this.container.parentNode.removeChild(this.container);
      this.map = undefined;
    }
  }

  // ---------------------------
  // 7) Create a GeoJSON circle (Polygon) for a given radius (miles)
  // ---------------------------
  function createGeoJSONCircle(center, radiusInMiles, points = 64) {
    if (!center || center.length !== 2) {
        console.error('Invalid center coordinates:', center);
        return null;
    }

    const coords = { lng: center[0], lat: center[1] };
    const km = radiusInMiles * 1.60934; // Convert miles to kilometers
    const ret = [];

    // Calculate distance using more precise conversion factors
    // These factors account for the Earth's ellipsoid shape
    const latRadian = coords.lat * Math.PI / 180;
    const metersPerDegreeLatitude = 111132.92 - 559.82 * Math.cos(2 * latRadian) + 1.175 * Math.cos(4 * latRadian);
    const metersPerDegreeLongitude = 111412.84 * Math.cos(latRadian) - 93.5 * Math.cos(3 * latRadian);

    const distanceX = (km * 1000) / metersPerDegreeLongitude;
    const distanceY = (km * 1000) / metersPerDegreeLatitude;

    // Use more points for smaller radii to make the circle smoother
    const adjustedPoints = radiusInMiles < 0.1 ? 128 : 
                         radiusInMiles < 0.5 ? 96 : 
                         points;

    for(let i = 0; i < adjustedPoints; i++) {
        const theta = (i / adjustedPoints) * (2 * Math.PI);
        const x = distanceX * Math.cos(theta);
        const y = distanceY * Math.sin(theta);

        // Ensure coordinates are within valid ranges
        const newLng = Math.max(-180, Math.min(180, coords.lng + x));
        const newLat = Math.max(-90, Math.min(90, coords.lat + y));
        
        ret.push([newLng, newLat]);
    }
    ret.push(ret[0]); // Close the circle

    return {
        type: "Feature",
        geometry: {
            type: "Polygon",
            coordinates: [ret]
        }
    };
  }

  // ---------------------------
  // 8) Remove radius circle from the map
  // ---------------------------
  function removeRadiusCircle() {
    if (!map) return;
    if (map.getLayer("radius-circle-fill")) {
      map.removeLayer("radius-circle-fill");
    }
    if (map.getLayer("radius-circle-outline")) {
      map.removeLayer("radius-circle-outline");
    }
    if (map.getSource("radius-circle")) {
      map.removeSource("radius-circle");
    }
  }

  // ---------------------------
  // 9) Main map update function
  // ---------------------------
  function updateMap() {
    if (!map) return;
    
    // Update markers
    loadPublicMementoMarkers();
    loadUserMementosOnMap();
    loadAllUserMementosOnMap();
    
    // Update marker sizes based on current zoom
    updateMarkerSizes(map.getZoom());

    // Update live location marker
    if (userLocation) {
      // Remove existing live location marker if it exists
      if (liveLocationMarker) {
        liveLocationMarker.remove();
      }

      // Create new live location marker
      const el = document.createElement('div');
      el.className = 'live-location-marker';
      
      // Create the dot
      const dot = document.createElement('div');
      dot.className = 'location-dot';
      el.appendChild(dot);
      
      // Create the pulse effect
      const pulse = document.createElement('div');
      pulse.className = 'location-pulse';
      el.appendChild(pulse);
      
      // Create the heading indicator
      const heading = document.createElement('div');
      heading.className = 'location-heading';
      el.appendChild(heading);

      // Create and add the marker to the map
      liveLocationMarker = new mapboxgl.Marker({
        element: el,
        anchor: 'center'
      })
        .setLngLat(userLocation)
        .addTo(map);
    }
  }

  // ---------------------------
  // 10) Geolocation: watchPosition
  // ---------------------------
  function getUserLocation() {
    if (navigator.geolocation) {
      // Clear any existing watch
      if (window.locationWatchId) {
        navigator.geolocation.clearWatch(window.locationWatchId);
      }

      // Get initial position with maximum accuracy
      navigator.geolocation.getCurrentPosition(
        position => {
          console.log('Initial position received:', position.coords);
          userLocation = [position.coords.longitude, position.coords.latitude];
          if (map) {
            map.flyTo({
              center: userLocation,
              zoom: 15, // Increased zoom for better accuracy visualization
              essential: true,
              duration: 1000
            });
            updateMap();
          }
        },
        error => {
          console.error("Initial Position Error:", error);
          showToast("Error getting location. Please enable location services and refresh the page.");
        },
        { 
          enableHighAccuracy: true,
          timeout: 5000, // Reduced timeout for faster response
          maximumAge: 0 // Always get fresh position
        }
      );

      // Watch for position updates with maximum accuracy
      const watchId = navigator.geolocation.watchPosition(
        position => {
          console.log('Position update received:', position.coords);
          const newLocation = [position.coords.longitude, position.coords.latitude];
          
          // Only update if position has changed significantly (more than 1 meter)
          if (!userLocation || 
              Math.abs(newLocation[0] - userLocation[0]) > 0.00001 || 
              Math.abs(newLocation[1] - userLocation[1]) > 0.00001) {
            
            userLocation = newLocation;
            if (map) {
              updateMap();
            }
          }
        },
        error => {
          console.error("Watch Position Error:", error);
          showToast("Error tracking location. Please check your location settings.");
        },
        { 
          enableHighAccuracy: true,
          timeout: 3000, // Reduced timeout for more frequent updates
          maximumAge: 1000 // Accept positions up to 1 second old
        }
      );

      window.locationWatchId = watchId;
    } else {
      console.error("Geolocation not supported");
      showToast("Geolocation is not supported by your browser");
    }
  }

  // ---------------------------
  // 11) Map Functions
  // ---------------------------
  function initializeMap(center = [-73.935242, 40.730610]) { // Center of NYC
    // Safely remove existing map if it exists
    if (map) {
        try {
            map.remove();
        } catch (error) {
            console.warn('Error removing map:', error);
        }
    }

    // Initialize map with custom style
    const mapStyle = 'mapbox://styles/0209vaibhav/cm9xf0p7u001301qxdqls09o9';
    initializeMapWithStyle(mapStyle, center);
  }

  function initializeMapWithStyle(mapStyle, center) {
    try {
        map = new mapboxgl.Map({
            container: 'map',
            style: mapStyle,
            center: [-73.935242, 40.730610], // Center of NYC
            zoom: 10, // Start with NYC overview
            minZoom: 8,
            maxZoom: 18,
            maxBounds: [
                [-74.5, 40.3], // Southwest corner (expanded)
                [-73.5, 41.1]  // Northeast corner (expanded)
            ],
            fog: {
                'range': [0.8, 8],
                'color': '#ffffff',
                'high-color': '#245cdf',
                'space-color': '#000000',
                'horizon-blend': 0.02,
                'star-intensity': 0.15
            }
        });

        // Add style change event listener
        map.on('style.load', () => {
            console.log('Map style loaded successfully');
            // Ensure map is properly sized after style load
            map.resize();

            // Check if source already exists before adding
            if (!map.getSource('mementos')) {
                map.addSource('mementos', {
                    type: 'geojson',
                    data: {
                        type: 'FeatureCollection',
                        features: []
                    }
                });
            }

            // Initialize radius circle by default
            const radiusToggle = document.getElementById('radius-toggle');
            const radiusSlider = document.getElementById('radius-slider');
            
            if (radiusToggle) {
                radiusToggle.checked = true;
            }
            
            if (radiusSlider) {
                const radiusMiles = parseFloat(radiusSlider.value);
                const center = userLocation || map.getCenter().toArray();
                updateRadiusCircle(center, radiusMiles);
                updateMarkersRadius();
            }
        });

        // Store map instance on the map element for global access
        const mapElement = document.getElementById('map');
        if (mapElement) {
            mapElement._mapboxgl_map = map;
        }

        // Add controls
        map.addControl(new mapboxgl.NavigationControl(), 'top-right');
        
        // Add custom locate control
        map.addControl(new LocateMeControl());

        // Add zoom event listener
        map.on('zoom', () => {
            updateMarkerSizes(map.getZoom());
        });

        // Add move event listener for cluster count labels
        map.on('move', () => {
            const labels = document.querySelectorAll('.cluster-count-label');
            labels.forEach(label => {
                const lng = parseFloat(label.dataset.lng);
                const lat = parseFloat(label.dataset.lat);
                if (!isNaN(lng) && !isNaN(lat)) {
                    const point = map.project([lng, lat]);
                    label.style.left = `${point.x}px`;
                    label.style.top = `${point.y}px`;
                }
            });
        });

        // Map is ready
        map.on('load', () => {
            // Map is initialized and ready
            if (map) {
                map.resize();
                // Load all types of markers after map is ready
                loadPublicMementoMarkers();
                loadUserMementosOnMap();
                loadAllUserMementosOnMap();
                // Set initial marker sizes
                updateMarkerSizes(map.getZoom());

                // Initialize radius circle
                const radiusToggle = document.getElementById('radius-toggle');
                const radiusSlider = document.getElementById('radius-slider');
                
                if (radiusToggle) {
                    radiusToggle.checked = true;
                }
                
                if (radiusSlider) {
                    const radiusMiles = parseFloat(radiusSlider.value);
                    const center = userLocation || map.getCenter().toArray();
                    updateRadiusCircle(center, radiusMiles);
                    updateMarkersRadius();
                }
            }
        });

    } catch (error) {
        console.error('Error initializing map:', error);
    }
  }

  // ---------------------------
  // 12) Initialize Default State
  // ---------------------------
  // Set up initial map
  mainContent.innerHTML = `<div id="map" style="width: 100%; height: 100vh;"></div>`;
  mapboxgl.accessToken = 'pk.eyJ1IjoiMDIwOXZhaWJoYXYiLCJhIjoiY2x6cW4xY2w5MWswZDJxcHhreHZ2OG5mbSJ9.ozamGsW5CZrZdL5bG7n_0A';
  
  // Request location before initializing map
  if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
          position => {
              userLocation = [position.coords.longitude, position.coords.latitude];
              // Wait for splash screen to complete before initializing map
              const splashScreen = document.querySelector('.splash-screen');
              if (splashScreen) {
                  splashScreen.addEventListener('animationend', () => {
                      // Initialize map centered on NYC after splash screen
                      initializeMap();
                      
                      // Then after 5 seconds, zoom to user location
                      setTimeout(() => {
                          map.flyTo({
                              center: userLocation,
                              zoom: 14,
                              duration: 5000,
                              essential: true
                          });
                          updateMap();
                      }, 5000); // Wait 5 seconds before zooming to location
                  });
              } else {
                  // If no splash screen, initialize immediately
                  initializeMap();
                  map.flyTo({
                      center: userLocation,
                      zoom: 14,
                      duration: 2000,
                      essential: true
                  });
                  updateMap();
              }
          },
          error => {
              console.error("Initial Position Error:", error);
              const splashScreen = document.querySelector('.splash-screen');
              if (splashScreen) {
                  splashScreen.addEventListener('animationend', () => {
                      initializeMap();
                      updateMap();
                  });
              } else {
                  initializeMap();
                  updateMap();
              }
          },
          { 
              enableHighAccuracy: true,
              timeout: 5000,
              maximumAge: 0
          }
      );
  } else {
      const splashScreen = document.querySelector('.splash-screen');
      if (splashScreen) {
          splashScreen.addEventListener('animationend', () => {
              initializeMap();
              updateMap();
          });
      } else {
          initializeMap();
          updateMap();
      }
  }

  // Set Discovery as active tab
  document.querySelectorAll('.activity-tab').forEach(tab => tab.classList.remove('active'));
  const discoveryTab = document.querySelector('[data-activity="discovery"]');
  if (discoveryTab) {
    discoveryTab.classList.add('active');
    currentTab = 'discovery';
  }

  // Hide all subtabs
  document.querySelectorAll('.explorer-subtabs').forEach(subtab => {
    subtab.classList.remove('show');
  });

  // Set left panel state to be expanded by default
  infoTab.classList.remove('hidden');
  collapseLeftBtn.classList.remove('hidden');
  expandLeftBtn.classList.add('hidden');

  // Resize map after panel state changes
  setTimeout(() => {
    if (map) {
      map.resize();
    }
  }, 300);

  // ---------------------------
  // 13) Event Listeners
  // ---------------------------
  // Function to initialize and update the map
  function initializeAndUpdateMap() {
    if (!map) {
      initializeMap();
    }
    if (userLocation) {
      map.flyTo({
        center: userLocation,
        zoom: 14,
        essential: true,
        duration: 3000,
        curve: 1.5
      });
      updateMap();
    }
    // Ensure map is properly sized
    setTimeout(() => {
      if (map) {
        map.resize();
      }
    }, 300);
  }

  // Add event listeners for filter group chevrons
  document.querySelectorAll('.filter-group h3 i.fas.fa-chevron-down').forEach(chevron => {
    chevron.addEventListener('click', function() {
      const filterGroup = this.closest('.filter-group');
      const radiusInteraction = filterGroup.querySelector('.radius-interaction-container');
      const sliderContainer = filterGroup.querySelector('.slider-container');
      
      // Toggle visibility of both containers
      [radiusInteraction, sliderContainer].forEach(container => {
        if (container) {
          container.style.display = container.style.display === 'none' ? 'block' : 'none';
        }
      });
      
      // Rotate chevron icon
      this.style.transform = this.style.transform === 'rotate(180deg)' ? 'rotate(0deg)' : 'rotate(180deg)';
    });
  });

  // Main activity tab click handler
  document.querySelectorAll('.activity-tab').forEach(tab => {
    tab.addEventListener('click', function(e) {
        e.preventDefault();
        
        // Hide filter settings container when switching main activities
        const filterSettings = document.querySelector('.filter-settings-container');
        if (filterSettings) {
            filterSettings.classList.add('hidden');
        }

        const activity = this.getAttribute('data-activity');
        if (currentTab === activity) return;
        currentTab = activity;

        // Hide settings container when switching main tabs
        const settingsContainer = document.getElementById('settings-container');
        if (settingsContainer) {
            settingsContainer.style.display = 'none';
        }

        // Hide all sub-tabs first
        document.querySelectorAll('.explorer-subtabs').forEach(subtab => {
          subtab.classList.remove('show');
        });
        
        // Show the sub-tabs for the clicked tab
        const subtabs = this.nextElementSibling;
        if (subtabs && subtabs.classList.contains('explorer-subtabs')) {
          subtabs.classList.add('show');
        }
        
        // Update tab states
        document.querySelectorAll('.activity-tab').forEach(t => 
          t.classList.remove('active'));
        this.classList.add('active');

        // Hide all tab content first
        document.querySelectorAll('.tab-content').forEach(content => {
          content.classList.add('hidden');
        });

        // Hide about content when switching to other tabs
        const aboutContent = document.querySelector('.about-content');
        if (aboutContent) {
          aboutContent.classList.add('hidden');
        }

        // Show the selected tab's content
        const tabContent = document.getElementById(activity);
        if (tabContent) {
          tabContent.classList.remove('hidden');
        }

        // If this is the journey tab, show the capture form by default
        if (activity === 'journey') {
          const captureForm = document.getElementById('journey-capture-form');
          if (captureForm) {
            captureForm.classList.remove('hidden');
          }
        }

        // Set default active subtabs based on activity
        const defaultSubtabs = {
          'explorer': 'profile',
          'journey': 'capture',
          'discovery': 'live-feed',
          'archive': 'favorites'
        };

        // Get all subtab buttons for the current activity
        const currentSubtabs = subtabs.querySelectorAll('.explorer-tab-btn');
        currentSubtabs.forEach(btn => {
          btn.classList.remove('active');
          if (btn.getAttribute('data-tab') === defaultSubtabs[activity]) {
            btn.classList.add('active');
            handleSubTabAction(activity, defaultSubtabs[activity]);
          }
        });

        // Initialize and update map for the current tab
        initializeAndUpdateMap();
      });
  });

  // ---------------------------
  // 14) Radius Toggle Handling
  // ---------------------------
  if (radiusToggle) {
    radiusToggle.addEventListener("change", function() {
      const listViewBtn = document.querySelector('[data-tab="list-view"]');
      if (listViewBtn && listViewBtn.classList.contains('active')) {
        handleSubTabAction('discovery', 'list-view');
      }
    });
  }

  if (radiusSlider) {
    radiusSlider.addEventListener("input", function() {
      const listViewBtn = document.querySelector('[data-tab="list-view"]');
      if (listViewBtn && listViewBtn.classList.contains('active')) {
        handleSubTabAction('discovery', 'list-view');
      }
    });
  }

  // ---------------------------
  // 15) Radius Slider Handling
  // ---------------------------
  if (radiusSlider && radiusValue) {
    radiusSlider.addEventListener("input", function () {
      const formattedValue = Number(this.value).toFixed(2);
      radiusValue.innerText = formattedValue + ' mi';
      
      if (userLocation) {
        updateRadiusCircle(userLocation, parseFloat(formattedValue));
      } else if (map) {
          const center = map.getCenter().toArray();
          updateRadiusCircle(center, parseFloat(formattedValue));
      }
      updateMap();
    });
  }

  // ---------------------------
  // 16) Panel Collapse/Expand with map.resize()
  // ---------------------------
  if (collapseLeftBtn) {
    // Remove any existing listeners
    const newCollapseBtn = collapseLeftBtn.cloneNode(true);
    collapseLeftBtn.parentNode.replaceChild(newCollapseBtn, collapseLeftBtn);
    
    // Add fresh event listeners
    newCollapseBtn.addEventListener("touchend", function(e) {
      e.preventDefault();
      e.stopPropagation();
      console.log("Collapse Left Button Touched");
      const infoTab = document.querySelector('.info-tab');
      if (infoTab) {
        infoTab.style.visibility = 'hidden';
        infoTab.classList.add('hidden');
        newCollapseBtn.classList.add('hidden');
        document.getElementById('expand-left').classList.remove('hidden');
        if (map) {
          setTimeout(() => { map.resize(); }, 300);
        }
      }
    }, { passive: false });

    newCollapseBtn.addEventListener("click", function(e) {
      e.preventDefault();
      e.stopPropagation();
      console.log("Collapse Left Button Clicked");
      const infoTab = document.querySelector('.info-tab');
      if (infoTab) {
        infoTab.style.visibility = 'hidden';
        infoTab.classList.add('hidden');
        newCollapseBtn.classList.add('hidden');
        document.getElementById('expand-left').classList.remove('hidden');
        if (map) {
          setTimeout(() => { map.resize(); }, 300);
        }
      }
    });
  }

  if (expandLeftBtn) {
    // Remove any existing listeners
    const newExpandBtn = expandLeftBtn.cloneNode(true);
    expandLeftBtn.parentNode.replaceChild(newExpandBtn, expandLeftBtn);
    
    // Add fresh event listeners
    newExpandBtn.addEventListener("touchend", function(e) {
      e.preventDefault();
      e.stopPropagation();
      console.log("Expand Left Button Touched");
      const infoTab = document.querySelector('.info-tab');
      if (infoTab) {
        infoTab.style.visibility = 'visible';
        infoTab.classList.remove('hidden');
        newExpandBtn.classList.add('hidden');
        document.getElementById('collapse-left').classList.remove('hidden');
        if (map) {
          setTimeout(() => { map.resize(); }, 300);
        }
      }
    }, { passive: false });

    newExpandBtn.addEventListener("click", function(e) {
      e.preventDefault();
      e.stopPropagation();
      console.log("Expand Left Button Clicked");
      const infoTab = document.querySelector('.info-tab');
      if (infoTab) {
        infoTab.style.visibility = 'visible';
        infoTab.classList.remove('hidden');
        newExpandBtn.classList.add('hidden');
        document.getElementById('collapse-left').classList.remove('hidden');
        if (map) {
          setTimeout(() => { map.resize(); }, 300);
        }
      }
    });
  }

  if (collapseRightBtn) {
    collapseRightBtn.addEventListener("click", function() {
      console.log("Collapse Right Button Clicked");
      rightPanel.classList.add("hidden");
      expandRightBtn.classList.remove("hidden");
      collapseRightBtn.classList.add("hidden");
      if (map) {
        setTimeout(() => { map.resize(); }, 300);
      }
    });
  }

  if (expandRightBtn) {
    expandRightBtn.addEventListener("click", function() {
      console.log("Expand Right Button Clicked");
      rightPanel.classList.remove("hidden");
      expandRightBtn.classList.add("hidden");
      collapseRightBtn.classList.remove("hidden");
      if (map) {
        setTimeout(() => { map.resize(); }, 300);
      }
    });
  }

  // Move handleTabClick inside DOMContentLoaded
  function handleTabClick(activity) {
    // Hide all sub-tabs first
    document.querySelectorAll('.explorer-subtabs').forEach(subtab => {
        subtab.classList.remove('show');
    });
    
    // Hide filter settings container when switching activity tabs
    const filterSettings = document.querySelector('.filter-settings-container');
    if (filterSettings) {
        filterSettings.classList.add('hidden');
    }
    
    // Show the sub-tabs for the clicked tab
    const subtabs = this.nextElementSibling;
    if (subtabs && subtabs.classList.contains('explorer-subtabs')) {
        subtabs.classList.add('show');
    }
    
    // Update tab states
    document.querySelectorAll('.activity-tab').forEach(t => 
      t.classList.remove('active'));
    this.classList.add('active');

    // Hide all tab content first
    document.querySelectorAll('.tab-content').forEach(content => {
      content.classList.add('hidden');
    });

    // Show the selected tab's content
    const tabContent = document.getElementById(activity);
    if (tabContent) {
      tabContent.classList.remove('hidden');
    }

    // Set default active subtabs based on activity
    const defaultSubtabs = {
      'explorer': 'profile',
      'journey': 'capture',
      'discovery': 'live-feed',
      'archive': 'favorites'
    };

    // Get all subtab buttons for the current activity
    const currentSubtabs = subtabs.querySelectorAll('.explorer-tab-btn');
    currentSubtabs.forEach(btn => {
      btn.classList.remove('active');
      if (btn.getAttribute('data-tab') === defaultSubtabs[activity]) {
        btn.classList.add('active');
        handleSubTabAction(activity, defaultSubtabs[activity]);
      }
    });

    // Initialize and update map for the current tab
    initializeAndUpdateMap();
    
    // Hide capture form when switching tabs
    handleCaptureFormVisibility(activity, '');
  }

  // Modify the handleSubTabAction function
  function handleSubTabAction(activity, tabId) {
    // Hide all containers by default
    const journeyCaptureForm = document.getElementById('journey-capture-form');
    const draftsContainer = document.getElementById('journey-drafts-container');
    const archiveContainer = document.getElementById('archive-container');
    const curatedContainer = document.querySelector('.curated-container');
    const liveFeedContainer = document.querySelector('.live-feed-container');
    const filterSettingsContainer = document.querySelector('.filter-settings-container');
    const authContainer = document.getElementById('auth-container');
    const aboutContent = document.querySelector('.about-content');
    const aboutCredits = document.getElementById('about-credits');
    const aboutInfo = document.getElementById('about-info');
    const myMementosContainer = document.querySelector('.my-mementos-container');
    const mementoInfoContent = document.getElementById('memento-info-content');

    if (journeyCaptureForm) journeyCaptureForm.classList.add('hidden');
    if (draftsContainer) draftsContainer.classList.add('hidden');
    if (archiveContainer) archiveContainer.style.display = 'none';
    if (curatedContainer) curatedContainer.classList.add('hidden');
    if (liveFeedContainer) liveFeedContainer.classList.add('hidden');
    if (filterSettingsContainer) filterSettingsContainer.classList.add('hidden');
    if (aboutContent) aboutContent.classList.add('hidden');
    if (aboutCredits) aboutCredits.classList.add('hidden');
    if (aboutInfo) aboutInfo.classList.add('hidden');
    if (myMementosContainer) myMementosContainer.classList.add('hidden');
    if (mementoInfoContent) {
      mementoInfoContent.classList.remove('active');
      mementoInfoContent.style.display = 'none';
    }

    // Show the appropriate container based on activity and tabId
    switch (activity) {
      case 'journey':
        if (tabId === 'capture' && journeyCaptureForm) {
          journeyCaptureForm.classList.remove('hidden');
        } else if (tabId === 'drafts' && draftsContainer) {
          draftsContainer.classList.remove('hidden');
        }
        break;
      case 'discovery':
        if (tabId === 'live-feed' && liveFeedContainer) {
          liveFeedContainer.classList.remove('hidden');
        } else if (tabId === 'filter' && filterSettingsContainer) {
          filterSettingsContainer.classList.remove('hidden');
        } else if (tabId === 'curated' && curatedContainer) {
          curatedContainer.classList.remove('hidden');
        }
        break;
      case 'archive':
        if (archiveContainer) {
          archiveContainer.style.display = 'block';
        }
        if (authContainer) {
          authContainer.classList.add('hidden');
        }
        break;
      case 'about':
        if (aboutContent) {
          aboutContent.classList.remove('hidden');
          if (tabId === 'credits' && aboutCredits) {
            aboutCredits.classList.remove('hidden');
          } else if (tabId === 'info' && aboutInfo) {
            aboutInfo.classList.remove('hidden');
          }
        }
        break;
      case 'explorer':
        if (tabId === 'my-mementos' && myMementosContainer) {
          myMementosContainer.classList.remove('hidden');
          if (authContainer) {
            authContainer.classList.add('hidden');
          }
        } else if (tabId === 'profile') {
          if (authContainer) {
            authContainer.classList.remove('hidden');
          }
        } else if (tabId === 'archive') {
          if (archiveContainer) {
            archiveContainer.style.display = 'block';
          }
          if (authContainer) {
            authContainer.classList.add('hidden');
          }
        }
        break;
    }

    // Initialize and update map for the current tab
    initializeAndUpdateMap();
  }

  // Add event listeners for sub-tab buttons
  document.querySelectorAll('.explorer-tab-btn').forEach(button => {
    button.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      
      // Find the closest .explorer-subtabs container
      const subtabsContainer = button.closest('.explorer-subtabs');
      if (!subtabsContainer) return;
      
      // Remove active class from all buttons in this container
      subtabsContainer.querySelectorAll('.explorer-tab-btn').forEach(btn => 
        btn.classList.remove('active'));
      
      // Add active class to clicked button
      button.classList.add('active');
      
      // Get the parent tab's activity type
      const parentTab = button.closest('li').querySelector('.activity-tab');
      if (!parentTab) return;
      
      const activity = parentTab.getAttribute('data-activity');
      const tabId = button.getAttribute('data-tab');
      
      // For category view only, expand right panel immediately
      if (activity === 'discovery' && tabId === 'live-map') {
        rightPanel.classList.remove('hidden');
        expandRightBtn.classList.add('hidden');
        collapseRightBtn.classList.remove('hidden');
        // Resize map after panel expansion
        if (map) {
          setTimeout(() => { map.resize(); }, 300);
        }
      }
      
      // Handle sub-tab specific actions
      handleSubTabAction(activity, tabId);
    });
  });

  // Add event listeners for explorer tab buttons
  document.querySelectorAll('.explorer-tab-btn').forEach(button => {
    button.addEventListener('click', function() {
      // Get the parent tab's activity type
      const parentTab = button.closest('li').querySelector('.activity-tab');
      if (!parentTab) return;
      
      const activity = parentTab.getAttribute('data-activity');
      const tabId = this.getAttribute('data-tab');

      // Remove active class from all buttons in this container
      const subtabsContainer = this.closest('.explorer-subtabs');
      if (subtabsContainer) {
        subtabsContainer.querySelectorAll('.explorer-tab-btn').forEach(btn => {
          btn.classList.remove('active');
        });
      }
      
      // Add active class to clicked button
      this.classList.add('active');
      
      // For category view only, expand right panel immediately
      if (activity === 'discovery' && tabId === 'live-map') {
        rightPanel.classList.remove('hidden');
        expandRightBtn.classList.add('hidden');
        collapseRightBtn.classList.remove('hidden');
        // Resize map after panel expansion
        if (map) {
          setTimeout(() => { map.resize(); }, 300);
        }
      }
      
      // Handle sub-tab specific actions
      handleSubTabAction(activity, tabId);
    });
  });

  // ---------------------------
  // 17) Journey Capture Functionality
  // ---------------------------
  function initializeJourneyCapture() {
    const captureForm = document.getElementById('journey-capture-form');
    const mediaInput = document.getElementById('activity-media');
    const mediaPreview = document.getElementById('media-preview');
    
    // Form state
    let selectedLocation = null;
    let uploadedFiles = [];
    let selectedTags = [];
    let selectedCategory = '';
    let selectedDuration = '';
    let mementoTimestamp = new Date();
    let recognition = null;

    // Initialize speech recognition
    if ('webkitSpeechRecognition' in window) {
      recognition = new webkitSpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';
      recognition.maxAlternatives = 1;

      // Add timeout handling
      let recognitionTimeout;
      const RECOGNITION_TIMEOUT = 5000; // 5 seconds

      recognition.onstart = function() {
        // Clear any existing timeout
        if (recognitionTimeout) {
          clearTimeout(recognitionTimeout);
        }
        // Set new timeout
        recognitionTimeout = setTimeout(() => {
          recognition.stop();
          showToast('Voice input timed out. Please try again.', 'warning');
        }, RECOGNITION_TIMEOUT);
      };

      recognition.onend = function() {
        // Clear timeout when recognition ends
        if (recognitionTimeout) {
          clearTimeout(recognitionTimeout);
        }
        // Remove recording class from all buttons
        document.querySelectorAll('.voice-input-btn').forEach(btn => {
          btn.classList.remove('recording');
        });
      };

      recognition.onresult = function(event) {
        const result = event.results[0][0].transcript;
        const activeInput = document.activeElement;
        if (activeInput && (activeInput.id === 'memento-name' || activeInput.id === 'memento-description')) {
          activeInput.value = result;
          // Trigger input event to update any listeners
          activeInput.dispatchEvent(new Event('input', { bubbles: true }));
          showToast('Voice input captured', 'success');
        }
      };

      recognition.onerror = function(event) {
        // Clear timeout on error
        if (recognitionTimeout) {
          clearTimeout(recognitionTimeout);
        }
        
        // Remove recording class from all buttons
        document.querySelectorAll('.voice-input-btn').forEach(btn => {
          btn.classList.remove('recording');
        });

        let errorMessage = 'Voice input error';
        switch(event.error) {
          case 'no-speech':
            errorMessage = 'No speech detected. Please try again.';
            break;
          case 'aborted':
            errorMessage = 'Voice input was aborted.';
            break;
          case 'audio-capture':
            errorMessage = 'No microphone detected. Please check your microphone.';
            break;
          case 'network':
            errorMessage = 'Network error occurred. Please check your connection.';
            break;
          case 'not-allowed':
            errorMessage = 'Microphone access was denied. Please allow microphone access.';
            break;
          case 'service-not-allowed':
            errorMessage = 'Speech recognition service is not allowed.';
            break;
          default:
            errorMessage = `Voice input error: ${event.error}`;
        }
        showToast(errorMessage, 'error');
      };
    }

    // Add voice input buttons to form fields
    const nameInput = document.getElementById('memento-name');
    const descriptionInput = document.getElementById('memento-description');

    if (nameInput) {
      const nameVoiceBtn = document.createElement('button');
      nameVoiceBtn.className = 'voice-input-btn';
      nameVoiceBtn.innerHTML = '<i class="fas fa-microphone"></i>';
      nameVoiceBtn.title = 'Use voice input';
      nameVoiceBtn.addEventListener('click', () => {
        if (recognition) {
          recognition.start();
          nameVoiceBtn.classList.add('recording');
          showToast('Listening...', 'info');
        } else {
          showToast('Voice input is not supported in this browser', 'error');
        }
      });
      nameInput.parentNode.insertBefore(nameVoiceBtn, nameInput.nextSibling);
    }

    if (descriptionInput) {
      const descriptionVoiceBtn = document.createElement('button');
      descriptionVoiceBtn.className = 'voice-input-btn';
      descriptionVoiceBtn.innerHTML = '<i class="fas fa-microphone"></i>';
      descriptionVoiceBtn.title = 'Use voice input';
      descriptionVoiceBtn.addEventListener('click', () => {
        if (recognition) {
          recognition.start();
          descriptionVoiceBtn.classList.add('recording');
          showToast('Listening...', 'info');
        } else {
          showToast('Voice input is not supported in this browser', 'error');
        }
      });
      descriptionInput.parentNode.insertBefore(descriptionVoiceBtn, descriptionInput.nextSibling);
    }

    // Add styles for voice input buttons
    const style = document.createElement('style');
    style.textContent = `
      .voice-input-btn {
        position: absolute;
        right: 10px;
        top: 50%;
        background: none;
        border: none;
        cursor: pointer;
        color: #666;
        padding: 5px;
        transition: color 0.3s ease;
      }
      .voice-input-btn:hover {
        color: #000;
      }
      .voice-input-btn.recording {
        color: #ff0000;
        animation: pulse 1.5s infinite;
      }
      @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
      }
      .form-group {
        position: relative;
      }
    `;
    document.head.appendChild(style);

    if (!captureForm || !mediaInput || !mediaPreview) {
        console.error('Some journey capture elements are missing');
        return;
    }
    
    // Function to populate tags
    function populateTags() {
      const tagSelect = document.getElementById('tag-select');
      if (!tagSelect) return;

      // Fetch tags from the JSON file
      fetch('memento_tags.json')
        .then(response => response.json())
        .then(tags => {
          tagSelect.innerHTML = '<option value="">Select tags...</option>' +
            tags.map(tag => `
              <option value="${tag.id}" data-symbol="${tag.symbol}">
                ${tag.symbol} ${tag.name}
              </option>
            `).join('');
        })
        .catch(error => {
          console.error('Error loading tags:', error);
          showToast('Error loading tags', 'error');
        });
    }

    // Function to populate categories
    function populateCategories() {
      const categorySelect = document.getElementById('category-select');
      if (!categorySelect) return;

      // Fetch categories from the JSON file
      fetch('memento_categories.json')
        .then(response => response.json())
        .then(categories => {
          categorySelect.innerHTML = '<option value="">Select category...</option>' +
            categories.map(category => `
              <option value="${category.id}" data-symbol="${category.symbol}">
                ${category.symbol} ${category.name}
              </option>
            `).join('');
        })
        .catch(error => {
          console.error('Error loading categories:', error);
          showToast('Error loading categories', 'error');
        });
    }

    // Function to populate duration options
    function populateDurations() {
      const durationSelect = document.getElementById('duration-select');
      if (!durationSelect) return;

      // Use durations from the loaded JSON file
      const durations = window.durations || [];

      durationSelect.innerHTML = '<option value="">Select duration...</option>' +
        durations.map(duration => `
          <option value="${duration.id}" data-symbol="${duration.symbol}">
            ${duration.symbol} ${duration.name}
          </option>
        `).join('');
    }

    // Function to add selected item
    function addSelectedItem(type, value, text, symbol) {
      console.log('Adding selected item:', { type, value, text, symbol });
      // Special handling for category to use the correct container ID
      const containerId = type === 'category' ? 'selected-categories' : `selected-${type}s`;
      const container = document.getElementById(containerId);
      console.log('Container for', type, ':', container);
      if (!container) {
        console.log('Container not found for', type);
        return;
      }

      // For category and duration, clear existing items first
      if (type === 'category' || type === 'duration') {
        container.innerHTML = '';
      } else {
        // For tags, check if item is already selected
        if (container.querySelector(`[data-value="${value}"]`)) {
          showToast(`${type} already selected`, 'warning');
          return;
        }
      }

      // Create selected item element
      let nameOnly = text;
      if (symbol && text.startsWith(symbol)) {
        nameOnly = text.slice(symbol.length).trim();
      }
      const item = document.createElement('div');
      item.className = 'selected-item';
      item.setAttribute('data-value', value);
      item.innerHTML = `
        <span class="item-symbol">${symbol}</span>
        <span class="item-name">${nameOnly}</span>
        <button class="remove-item" title="Remove ${type}">
          <i class="fas fa-times"></i>
        </button>
      `;

      // Add remove functionality
      const removeBtn = item.querySelector('.remove-item');
      removeBtn.addEventListener('click', () => {
        item.remove();
        // Update the select to allow selecting this item again
        const select = document.getElementById(`${type}-select`);
        if (select) {
          const option = select.querySelector(`option[value="${value}"]`);
          if (option) option.disabled = false;
        }
      });

      container.appendChild(item);
      console.log('Added item to container:', item);

      // Disable the option in the select
      const select = document.getElementById(`${type}-select`);
      if (select) {
        const option = select.querySelector(`option[value="${value}"]`);
        if (option) option.disabled = true;
      }
    }

    // Initialize add selection buttons
    const addButtons = document.querySelectorAll('.add-selection-btn');
    addButtons.forEach(button => {
      button.addEventListener('click', () => {
        const type = button.dataset.type;
        const select = document.getElementById(`${type}-select`);
        if (!select || !select.value) {
          showToast(`Please select a ${type} first`, 'warning');
          return;
        }

        const option = select.options[select.selectedIndex];
        addSelectedItem(
          type,
          option.value,
          option.text,
          option.dataset.symbol
        );
      });
    });

    // Initialize selects with change event listeners
    function initializeSelects() {
      // Initialize tag select
      const tagSelect = document.getElementById('tag-select');
      if (tagSelect) {
        tagSelect.addEventListener('change', () => {
          if (tagSelect.value) {
            const option = tagSelect.options[tagSelect.selectedIndex];
            addSelectedItem(
              'tag',
              option.value,
              option.text,
              option.dataset.symbol
            );
            // Reset select to default option
            tagSelect.value = '';
          }
        });
      }

      // Initialize category select
      const categorySelect = document.getElementById('category-select');
      if (categorySelect) {
        console.log('Category select found:', categorySelect);
        categorySelect.addEventListener('change', (event) => {
          console.log('Category select changed:', event.target.value);
          if (categorySelect.value) {
            const option = categorySelect.options[categorySelect.selectedIndex];
            console.log('Selected category option:', option);
            // Remove any existing category
            const container = document.getElementById('selected-categories');
            console.log('Category container:', container);
            if (container) {
              container.innerHTML = '';
              addSelectedItem(
                'category',
                option.value,
                option.text,
                option.dataset.symbol
              );
              // Reset select to default option
              categorySelect.value = '';
            } else {
              console.error('Selected categories container not found');
              showToast('Error: Could not add category', 'error');
            }
          }
        });
      } else {
        console.log('Category select not found');
      }

      // Initialize duration select
      const durationSelect = document.getElementById('duration-select');
      if (durationSelect) {
        durationSelect.addEventListener('change', () => {
          if (durationSelect.value) {
            const option = durationSelect.options[durationSelect.selectedIndex];
            // Remove any existing duration
            const container = document.getElementById('selected-durations');
            if (container) {
              container.innerHTML = '';
            }
            addSelectedItem(
              'duration',
              option.value,
              option.text,
              option.dataset.symbol
            );
            // Reset select to default option
            durationSelect.value = '';
          }
        });
      }
    }

    // Populate all selects and initialize them
    async function initializeFormSelects() {
      // Wait for categories, tags, and durations to be loaded
      if (!window.durations) {
        console.log('Waiting for durations to load...');
        await new Promise(resolve => {
          const checkDurations = setInterval(() => {
            if (window.durations) {
              clearInterval(checkDurations);
              resolve();
            }
          }, 100);
        });
      }

      populateTags();
      populateCategories();
      populateDurations();
      initializeSelects();
    }

    // Initialize form selects
    initializeFormSelects();

    // Remove the add buttons since they're no longer needed
    document.querySelectorAll('.add-selection-btn').forEach(button => button.remove());

    // Function to collect memento data
    function collectMementoData() {
      const name = document.getElementById('memento-name').value.trim();
      const description = document.getElementById('memento-description').value.trim();
      
      // Get selected tags
      const selectedTags = Array.from(document.querySelectorAll('#selected-tags .selected-item'))
        .map(item => item.dataset.value);
      
      // Get selected category
      const selectedCategory = document.querySelector('#selected-categories .selected-item')?.dataset.value;
      
      // Get selected duration
      const selectedDuration = document.querySelector('#selected-durations .selected-item')?.dataset.value;

      // Validate location data
      let locationData = null;
      if (selectedLocation && selectedLocation.coordinates) {
        const { lat, lng } = selectedLocation.coordinates;
        if (typeof lat === 'number' && typeof lng === 'number' && !isNaN(lat) && !isNaN(lng)) {
          locationData = {
            address: selectedLocation.address || '',
            coordinates: {
              latitude: lat,
              longitude: lng
            }
          };
        }
      }

      return {
        name,
        description,
        tags: selectedTags,
        category: selectedCategory,
        location: locationData,
        timestamp: document.getElementById('memento-timestamp').value,
        duration: selectedDuration
      };
    }

    // Initialize step navigation
    const steps = document.querySelectorAll('.form-step');
    const progressSteps = document.querySelectorAll('.progress-steps .step');
    const nextButtons = document.querySelectorAll('.next-step-btn');
    const prevButtons = document.querySelectorAll('.prev-step-btn');

    // Function to show a specific step
    function showStep(stepNumber) {
      // Hide all steps
      steps.forEach(step => step.classList.remove('active'));
      progressSteps.forEach(step => step.classList.remove('active'));

      // Show the selected step
      const currentStep = document.querySelector(`.form-step[data-step="${stepNumber}"]`);
      const currentProgressStep = document.querySelector(`.progress-steps .step[data-step="${stepNumber}"]`);
      
      if (currentStep) currentStep.classList.add('active');
      if (currentProgressStep) currentProgressStep.classList.add('active');

      // Update progress bar
      progressSteps.forEach((step, index) => {
        if (index + 1 < stepNumber) {
          step.classList.add('completed');
        } else {
          step.classList.remove('completed');
        }
      });

      // If moving to preview step (step 5), update the preview
      if (stepNumber === '5') {
        updatePreview();
      }
    }

    // Function to update the preview
    function updatePreview() {
      // Get all the form data
      const name = document.getElementById('memento-name').value;
      const description = document.getElementById('memento-description').value;
      const location = document.getElementById('memento-location').value;
      const timestamp = document.getElementById('memento-timestamp').value;
      
      // Get current user
      const user = firebase.auth().currentUser;
      let username = 'Anonymous';
      if (user) {
        // Try to get username from user data
        firebase.firestore().collection('users').doc(user.uid).get()
          .then(doc => {
            if (doc.exists) {
              const userData = doc.data();
              username = userData.username || userData.displayName || user.email || 'Anonymous';
              // Update the author display
              document.getElementById('preview-user').textContent = username;
            }
          })
          .catch(error => {
            console.error('Error fetching username:', error);
            document.getElementById('preview-user').textContent = user.displayName || user.email || 'Anonymous';
          });
      }
      
      // Get selected category
      const categoryElement = document.querySelector('#selected-categories .selected-item');
      const category = categoryElement ? {
        symbol: categoryElement.querySelector('.item-symbol').textContent,
        name: categoryElement.querySelector('.item-name').textContent
      } : null;

      // Get selected tags
      const tagElements = document.querySelectorAll('#selected-tags .selected-item');
      const tags = Array.from(tagElements).map(tag => ({
        symbol: tag.querySelector('.item-symbol').textContent,
        name: tag.querySelector('.item-name').textContent
      }));

      // Get selected duration
      const durationElement = document.querySelector('#selected-durations .selected-item');
      const duration = durationElement ? {
        symbol: durationElement.querySelector('.item-symbol').textContent,
        name: durationElement.querySelector('.item-name').textContent
      } : null;

      // Update preview elements
      document.getElementById('preview-name').textContent = name || 'Untitled Memento';
      document.getElementById('preview-description').textContent = description || '';
      document.getElementById('preview-location').textContent = location || '';
      document.getElementById('preview-timestamp').textContent = timestamp ? formatDateTime(new Date(timestamp)) : '';
      document.getElementById('preview-duration').textContent = duration ? duration.name : '';
      
      // Update category
      const categoryPreview = document.getElementById('preview-category');
      if (category) {
        categoryPreview.innerHTML = `
          <span class="category-symbol">${category.symbol}</span>
          <span class="category-name">${category.name}</span>
        `;
      } else {
        categoryPreview.innerHTML = '';
      }

      // Update tags
      const tagsPreview = document.getElementById('preview-tags');
      if (tags.length > 0) {
        tagsPreview.innerHTML = tags.map(tag => `
          <span class="tag">
            <span class="tag-symbol">${tag.symbol}</span>
            <span class="tag-name">${tag.name}</span>
          </span>
        `).join('');
      } else {
        tagsPreview.innerHTML = '';
      }

      // Update media preview
      const mediaPreview = document.querySelector('.memento-preview-container .memento-media');
      if (uploadedFiles && uploadedFiles.length > 0) {
        mediaPreview.innerHTML = '';
        uploadedFiles.forEach(file => {
          const reader = new FileReader();
          reader.onload = (e) => {
            if (file.type.startsWith('image/')) {
              const img = document.createElement('img');
              img.src = e.target.result;
              img.alt = 'Memento media';
              mediaPreview.appendChild(img);
            } else if (file.type.startsWith('video/')) {
              const video = document.createElement('video');
              video.src = e.target.result;
              video.controls = true;
              mediaPreview.appendChild(video);
            }
          };
          reader.readAsDataURL(file);
        });
      } else {
        mediaPreview.innerHTML = '<div class="placeholder-media"><i class="fas fa-image"></i></div>';
      }
    }

    // Handle next button clicks
    nextButtons.forEach(button => {
      button.addEventListener('click', () => {
        const currentStep = button.closest('.form-step');
        const nextStepNumber = button.dataset.next;
        
        // Validate current step before proceeding
        if (validateStep(currentStep.dataset.step)) {
          showStep(nextStepNumber);
        }
      });
    });

    // Handle previous button clicks
    prevButtons.forEach(button => {
      button.addEventListener('click', () => {
        const prevStepNumber = button.dataset.prev;
        showStep(prevStepNumber);
      });
    });

    // Validate each step
    function validateStep(stepNumber) {
      const isEditMode = document.getElementById('save-memento-btn').textContent === 'Update Memento';
      
      switch (stepNumber) {
        case '1': // Media
          if (!isEditMode && (!uploadedFiles || uploadedFiles.length === 0)) {
            showToast('Please add at least one photo or video', 'error');
            return false;
          }
          return true;

        case '2': // Name and Description
          const name = document.getElementById('memento-name').value.trim();
          if (!name) {
            showToast('Please add a name for your memento', 'error');
            return false;
          }
          return true;

        case '3': // Tags and Categories
          const selectedTags = document.querySelectorAll('#selected-tags .selected-item');
          const selectedCategory = document.querySelector('#selected-categories .selected-item');
          const selectedDuration = document.querySelector('#selected-durations .selected-item');
          
          if (selectedTags.length === 0) {
            showToast('Please select at least one tag', 'error');
            return false;
          }
          if (!selectedCategory) {
            showToast('Please select a category', 'error');
            return false;
          }
          if (!selectedDuration) {
            showToast('Please select a duration', 'error');
            return false;
          }
          return true;

        case '4': // Location and Timestamp
          const location = document.getElementById('memento-location').value.trim();
          if (!location) {
            showToast('Please add a location', 'error');
            return false;
          }
          return true;

        default:
          return true;
      }
    }
    
    // Handle media upload
    if (mediaInput) {
      mediaInput.addEventListener('change', (event) => {
        const files = event.target.files;
        if (files.length > 0) {
          handleFileUpload(files);
        }
      });
      
      // Drag and drop functionality
      const mediaUploadBox = document.getElementById('media-upload-box');
      if (mediaUploadBox) {
        mediaUploadBox.addEventListener('dragover', (event) => {
          event.preventDefault();
          mediaUploadBox.classList.add('drag-over');
        });
        
        mediaUploadBox.addEventListener('dragleave', () => {
          mediaUploadBox.classList.remove('drag-over');
        });
        
        mediaUploadBox.addEventListener('drop', (event) => {
          event.preventDefault();
          mediaUploadBox.classList.remove('drag-over');
          
          const files = event.dataTransfer.files;
          if (files.length > 0) {
            handleFileUpload(files);
          }
        });
      }
    }
    
    // Handle file upload
    function handleFileUpload(files) {
      uploadedFiles = Array.from(files);
      const mediaPreview = document.getElementById('media-preview');
      
      if (!mediaPreview) {
        console.error('Media preview container not found');
        return;
      }
      
        mediaPreview.innerHTML = '';
        
        uploadedFiles.forEach((file, index) => {
            const reader = new FileReader();
        
            reader.onload = (e) => {
            const previewItem = document.createElement('div');
            previewItem.className = 'media-preview-item';
            
            if (file.type.startsWith('image/')) {
              const img = document.createElement('img');
              img.src = e.target.result;
            img.alt = 'Memento media';
              previewItem.appendChild(img);
            } else if (file.type.startsWith('video/')) {
              const video = document.createElement('video');
              video.src = e.target.result;
              video.controls = true;
              previewItem.appendChild(video);
            }
          
          // Add remove button
          const removeButton = document.createElement('button');
          removeButton.className = 'remove-media';
          removeButton.innerHTML = '<i class="fas fa-times"></i>';
          removeButton.onclick = (e) => {
            e.stopPropagation(); // Prevent event from bubbling up
            uploadedFiles.splice(index, 1);
            previewItem.remove();
            
            // If no files left, update the preview text
            if (uploadedFiles.length === 0) {
              const mediaHint = document.querySelector('.media-hint');
              if (mediaHint) {
                mediaHint.textContent = 'Tap to add photos or videos';
              }
            }
          };
          
          previewItem.appendChild(removeButton);
                mediaPreview.appendChild(previewItem);
            };
        
        reader.onerror = (error) => {
          console.error('Error reading file:', error);
          showToast('Error reading file', 'error');
        };
        
            reader.readAsDataURL(file);
        });
      
      // Update the hint text
      const mediaHint = document.querySelector('.media-hint');
      if (mediaHint) {
        mediaHint.textContent = `${uploadedFiles.length} media items selected`;
      }
    }

    // Initialize Google Places Autocomplete
    const locationInput = document.getElementById('memento-location');
    if (locationInput) {
      const autocomplete = new google.maps.places.Autocomplete(locationInput);
      autocomplete.addListener('place_changed', () => {
        const place = autocomplete.getPlace();
        if (place.geometry) {
          selectedLocation = {
            address: place.formatted_address,
            coordinates: {
              lat: place.geometry.location.lat(),
              lng: place.geometry.location.lng()
            }
          };
        }
      });
    }

    // Current location button
    const currentLocationBtn = document.getElementById('use-current-location-btn');
    if (currentLocationBtn) {
      currentLocationBtn.addEventListener('click', () => {
        if (navigator.geolocation) {
          currentLocationBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
          currentLocationBtn.disabled = true;
          
          navigator.geolocation.getCurrentPosition(
            (position) => {
              const lat = position.coords.latitude;
              const lng = position.coords.longitude;
              
              const geocoder = new google.maps.Geocoder();
              geocoder.geocode({ location: { lat, lng } }, (results, status) => {
                if (status === 'OK' && results[0]) {
                  locationInput.value = results[0].formatted_address;
                  selectedLocation = {
                    address: results[0].formatted_address,
                    coordinates: { lat, lng }
                  };
                } else {
                  console.error('Geocoder failed:', status);
                  showToast('Could not find address for this location', 'error');
                  
                  const fallbackAddress = `Location at ${lat.toFixed(6)}, ${lng.toFixed(6)}`;
                  locationInput.value = fallbackAddress;
                  selectedLocation = {
                    address: fallbackAddress,
                    coordinates: { lat, lng }
                  };
                }
                
                currentLocationBtn.innerHTML = '<i class="fas fa-crosshairs"></i>';
                currentLocationBtn.disabled = false;
              });
            },
            (error) => {
              console.error('Geolocation error:', error);
              let errorMessage = 'Error getting location';
              switch(error.code) {
                case error.PERMISSION_DENIED:
                  errorMessage = 'Location permission denied';
          break;
                case error.POSITION_UNAVAILABLE:
                  errorMessage = 'Location information unavailable';
          break;
                case error.TIMEOUT:
                  errorMessage = 'Location request timed out';
                  break;
              }
              showToast(errorMessage, 'error');
              currentLocationBtn.innerHTML = '<i class="fas fa-crosshairs"></i>';
              currentLocationBtn.disabled = false;
            },
            {
              enableHighAccuracy: true,
              timeout: 10000,
              maximumAge: 0
            }
          );
        } else {
          showToast('Geolocation is not supported by this browser', 'error');
        }
      });
    }

    // Current time button
    const currentTimeBtn = document.getElementById('use-current-time-btn');
    if (currentTimeBtn) {
      currentTimeBtn.addEventListener('click', () => {
        const now = new Date();
        mementoTimestamp = now;
        const mementoTimestampInput = document.getElementById('memento-timestamp');
        if (mementoTimestampInput) {
          mementoTimestampInput.value = formatDateTimeForInput(now);
        }
      });
    }

    // Handle save button
    const saveBtn = document.getElementById('save-memento-btn');
    if (saveBtn) {
      saveBtn.addEventListener('click', async () => {
        // Validate all steps
        for (let i = 1; i <= 4; i++) {
          if (!validateStep(i)) {
            showStep(i);
          return;
          }
        }
        
        try {
          const mementoData = collectMementoData();
          
          // Save as published memento
          const user = firebase.auth().currentUser;
          if (!user) {
            showToast('You need to be logged in to save mementos', 'error');
            return;
          }

          // Upload media files first
          const mediaUrls = await uploadMediaFiles(uploadedFiles, user.uid);
          mementoData.media = mediaUrls;
          
          // Save memento to Firestore
          const mementoId = await saveMemento(mementoData, user);
          
          showToast('Memento saved successfully!', 'success');
          
          // Reset form
          resetMementoForm();
          showStep(1);
          
          // Clear uploaded files array
          uploadedFiles = [];

          // Refresh markers and list view
          await loadUserMementosOnMap();
          await loadAllUserMementosOnMap();
          
          // Collapse the info panel (left panel)
          const infoTab = document.querySelector('.info-tab');
          const expandLeftBtn = document.getElementById('expand-left');
          const collapseLeftBtn = document.getElementById('collapse-left');
          if (infoTab && expandLeftBtn && collapseLeftBtn) {
            infoTab.style.visibility = 'hidden';
            infoTab.classList.add('hidden');
            expandLeftBtn.classList.remove('hidden');
            collapseLeftBtn.classList.add('hidden');
            if (map) {
              setTimeout(() => { map.resize(); }, 300);
            }
          }

          // Fly to the memento location on the map
          if (mementoData.location && mementoData.location.coordinates) {
            const { latitude, longitude } = mementoData.location.coordinates;
            if (typeof latitude === 'number' && typeof longitude === 'number') {
              map.flyTo({ center: [longitude, latitude], zoom: 15, essential: true });
              
              // Retrieve the newly created memento with the Firestore ID for the marker
              if (mementoId) {
                const newMementoDoc = await firebase.firestore().collection('mementos').doc(mementoId).get();
                if (newMementoDoc.exists) {
                  // Highlight the new marker if found
                  setTimeout(() => {
                    const newMarker = window.markers.find(m => m.mementoId === mementoId);
                    if (newMarker && newMarker.getElement) {
                      // Add highlight class
                      newMarker.getElement().classList.add('highlighted-marker');
                      
                      // Remove highlight after a delay
                      setTimeout(() => {
                        newMarker.getElement().classList.remove('highlighted-marker');
                      }, 3000);
                    }
                  }, 1000); // Delay to ensure markers are loaded
                }
              }
            }
          }
        } catch (error) {
          console.error('Error saving memento:', error);
          showToast(`Error saving memento: ${error.message}`, 'error');
        }
      });
    }
    
    // Function to reset the form
    function resetMementoForm() {
      // Reset name
      document.getElementById('memento-name').value = '';
      
      // Reset description
      document.getElementById('memento-description').value = '';
      
      // Reset tags
      const selectedTagsContainer = document.getElementById('selected-tags');
      if (selectedTagsContainer) selectedTagsContainer.innerHTML = '';
      const tagSelect = document.getElementById('tag-select');
      if (tagSelect) {
        tagSelect.value = '';
        // Reset any selected and disabled options
        const tagOptions = tagSelect.querySelectorAll('option');
        tagOptions.forEach(option => {
          option.selected = false;
          option.disabled = false;
        });
      }
      
      // Reset category
      const selectedCategoriesContainer = document.getElementById('selected-categories');
      if (selectedCategoriesContainer) selectedCategoriesContainer.innerHTML = '';
      const categorySelect = document.getElementById('category-select');
      if (categorySelect) {
        categorySelect.value = '';
        // Reset any selected and disabled options
        const categoryOptions = categorySelect.querySelectorAll('option');
        categoryOptions.forEach(option => {
          option.selected = false;
          option.disabled = false;
        });
      }
      
      // Reset duration
      const selectedDurationsContainer = document.getElementById('selected-durations');
      if (selectedDurationsContainer) selectedDurationsContainer.innerHTML = '';
      const durationSelect = document.getElementById('duration-select');
      if (durationSelect) {
        durationSelect.value = '';
        // Reset any selected and disabled options
        const durationOptions = durationSelect.querySelectorAll('option');
        durationOptions.forEach(option => {
          option.selected = false;
          option.disabled = false;
        });
      }
      
      // Reset location
      document.getElementById('memento-location').value = '';
      
      // Reset timestamp
      document.getElementById('memento-timestamp').value = '';
      
      // Reset media files
      const mediaPreview = document.getElementById('media-preview');
      if (mediaPreview) mediaPreview.innerHTML = '';
      const mediaInput = document.getElementById('activity-media');
      if (mediaInput) mediaInput.value = '';
      
      // Reset to step 1
      showStep(1);
    }

    // Initialize the form with step 1
    showStep(1);

    // Function to upload media files to Firebase Storage
    async function uploadMediaFiles(files, userId) {
      if (!files || files.length === 0) return [];
      
      const mediaUrls = await Promise.all(files.map(async file => {
        const fileType = file.type.split('/')[0]; // 'image' or 'video'
        const fileExt = file.name.split('.').pop();
        const fileName = `${Date.now()}_${Math.random().toString(36).substring(2, 15)}.${fileExt}`;
        
        // Create a copy of the file to prevent modification during upload
        const fileCopy = new File([file], file.name, { type: file.type });
        
        // Update storage reference path to match storage rules
        const storageRef = firebase.storage().ref(`mementos/${userId}/${fileName}`);
        
        try {
          // Add metadata to the upload
          const metadata = {
            contentType: file.type,
            customMetadata: {
              uploadedBy: userId,
              uploadedAt: new Date().toISOString(),
              type: fileType
            }
          };

          // Upload the file copy with metadata
          const snapshot = await storageRef.put(fileCopy, metadata);
          const downloadURL = await snapshot.ref.getDownloadURL();
          
          return {
            url: downloadURL,
            type: fileType,
            fileName: fileName
          };
        } catch (error) {
          console.error('Error uploading file:', error);
          if (error.code === 'storage/unauthorized') {
            throw new Error('You need to be logged in to upload files');
          } else if (error.code === 'storage/canceled') {
            throw new Error('Upload was canceled');
          } else if (error.code === 'storage/unknown') {
            throw new Error('An unknown error occurred during upload');
          } else {
            throw new Error(`Failed to upload ${fileType} file: ${error.message}`);
          }
        }
      }));
      
      return mediaUrls;
    }
    
    // Function to save memento to Firestore
    async function saveMemento(mementoData, user) {
      try {
        // Validate location data
        if (!mementoData.location || !mementoData.location.coordinates) {
          throw new Error('Location is required');
        }

        const { latitude, longitude } = mementoData.location.coordinates;
        if (typeof latitude !== 'number' || typeof longitude !== 'number' || 
            isNaN(latitude) || isNaN(longitude)) {
          throw new Error('Invalid location coordinates');
        }

        // Add user ID and timestamp
        const mementoWithMetadata = {
          ...mementoData,
          userId: user.uid,
          createdAt: firebase.firestore.FieldValue.serverTimestamp(),
          updatedAt: firebase.firestore.FieldValue.serverTimestamp()
        };

        // Save to Firestore
        const docRef = await firebase.firestore()
          .collection('mementos')
          .add(mementoWithMetadata);

        // Update the document with its ID
        await docRef.update({
          id: docRef.id
        });

        return docRef.id;
      } catch (error) {
        console.error('Error saving memento:', error);
        throw new Error(`Failed to save memento: ${error.message}`);
      }
    }

    // Restore Save as Draft and Cancel button event listeners
    const saveDraftBtn = document.getElementById('save-draft-btn');
    if (saveDraftBtn) {
      saveDraftBtn.addEventListener('click', async () => {
        try {
          const user = firebase.auth().currentUser;
          if (!user) {
            showToast('You need to be logged in to save drafts', 'error');
            return;
          }
          const draftData = collectMementoData();
          
          // Upload media files first if there are any
          if (uploadedFiles && uploadedFiles.length > 0) {
            draftData.media = await uploadMediaFiles(uploadedFiles, user.uid);
          } else {
            draftData.media = [];
          }
          
          // Add metadata
          draftData.createdAt = firebase.firestore.FieldValue.serverTimestamp();
          draftData.updatedAt = firebase.firestore.FieldValue.serverTimestamp();
          draftData.userId = user.uid;
          
          // Save to Firestore
          await firebase.firestore().collection('memento_drafts').add(draftData);
          showToast('Draft saved successfully!', 'success');
          resetMementoForm();
          showStep(1);

          // Switch to drafts tab and activate drafts subtab
          switchExplorerTab('journey');
          
          // Activate drafts subtab
          const draftsSubtab = document.querySelector('.explorer-tab-btn[data-tab="drafts"]');
          if (draftsSubtab) {
            // Remove active class from all subtabs in the journey tab
            const journeySubtabs = document.querySelector('.activity-tab[data-activity="journey"] + .explorer-subtabs');
            if (journeySubtabs) {
              journeySubtabs.querySelectorAll('.explorer-tab-btn').forEach(btn => {
                btn.classList.remove('active');
              });
            }
            // Add active class to drafts subtab
            draftsSubtab.classList.add('active');
            
            // Show drafts container
            const draftsContainer = document.getElementById('journey-drafts-container');
            if (draftsContainer) {
              draftsContainer.classList.remove('hidden');
              await initializeDraftsContainer();
            }
          }
        } catch (error) {
          console.error('Error saving draft:', error);
          showToast('Error saving draft', 'error');
        }
      });
    }

    const cancelBtn = document.getElementById('cancel-memento-btn');
    if (cancelBtn) {
      cancelBtn.addEventListener('click', () => {
        // Reset the form
        resetMementoForm();
        
        // Return to step 1
        showStep(1);
        
        // Show a toast message
        showToast('Memento creation cancelled', 'info');
      });
    }
  }

// ---------------------------
// 18) Geospatial utility functions
// ---------------------------
  // Function to calculate if a point is within radius
  function isPointWithinRadius(point, center, radiusMiles) {
    if (!center) return true; // If no center point, consider all points within radius
    
    // Convert coordinates to radians
    const lat1 = center[1] * Math.PI / 180;
    const lon1 = center[0] * Math.PI / 180;
    const lat2 = point[1] * Math.PI / 180;
    const lon2 = point[0] * Math.PI / 180;

    // Haversine formula
    const R = 3959; // Earth's radius in miles
    const dLat = lat2 - lat1;
    const dLon = lon2 - lon1;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
            Math.cos(lat1) * Math.cos(lat2) *
            Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    const distance = R * c;

    return distance <= radiusMiles;
  }

  // Function to load and display user mementos on the map
  async function loadUserMementosOnMap() {
    await loadMementosWithRetry(async () => {
      try {
        // Clear existing markers
        if (window.markers) {
          window.markers.forEach(marker => marker.remove());
        }
        window.markers = [];

        // Get current user
        const user = firebase.auth().currentUser;
        if (!user) {
          console.log('No user logged in');
          return;
        }

        // Get mementos from Firestore
        const mementosSnapshot = await firebase.firestore()
          .collection('mementos')
          .where('userId', '==', user.uid)
          .get();

        // Create markers for each memento
        mementosSnapshot.forEach((doc) => {
          try {
          const memento = { id: doc.id, ...doc.data() };
          
            // Only create marker if memento has valid location coordinates
            if (memento.location && 
                memento.location.coordinates && 
                typeof memento.location.coordinates.latitude === 'number' && 
                typeof memento.location.coordinates.longitude === 'number' &&
                !isNaN(memento.location.coordinates.latitude) && 
                !isNaN(memento.location.coordinates.longitude)) {
              
            // Create marker element
            const el = document.createElement('div');
            el.className = 'user-memento-marker';
            
            // Set data-marker-type attribute for filtering
            el.setAttribute('data-marker-type', 'user-memento');
            
            // Set background image if available
            if (memento.media && memento.media.length > 0) {
              const firstMedia = memento.media[0];
              const mediaUrl = typeof firstMedia === 'string' ? firstMedia : firstMedia.url;
              const mediaType = typeof firstMedia === 'string' ? 'image' : firstMedia.type;
              
              if (mediaType === 'video' || mediaUrl.endsWith('.mp4') || mediaUrl.endsWith('.webm')) {
                // For videos, create a video element
                const video = document.createElement('video');
                video.src = mediaUrl;
                video.muted = true;
                video.loop = true;
                video.playsInline = true;
                video.style.width = '100%';
                video.style.height = '100%';
                video.style.objectFit = 'cover';
                el.appendChild(video);
                
                // Start playing when marker is added
                video.play().catch(error => {
                  console.log('Video autoplay failed:', error);
                });
              } else {
                // For images, use background image
                el.style.backgroundImage = `url(${mediaUrl})`;
              }
            }
            
            // Create marker
            const marker = new mapboxgl.Marker({
              element: el,
              anchor: 'bottom'
            })
                .setLngLat([
                  memento.location.coordinates.longitude,
                  memento.location.coordinates.latitude
                ])
              .addTo(map);

              // Store memento ID with marker
              marker.mementoId = memento.id;
              
              // Add click handler to display memento in live feed
              marker.getElement().addEventListener('click', () => {
                // Only expand panel on mobile devices
                if (window.innerWidth < 1024) {
                  const infoTab = document.querySelector('.info-tab');
                  const expandLeftBtn = document.getElementById('expand-left');
                  const collapseLeftBtn = document.getElementById('collapse-left');
                  
                  if (infoTab && expandLeftBtn && collapseLeftBtn) {
                    infoTab.style.visibility = 'visible';
                    infoTab.classList.remove('hidden');
                    expandLeftBtn.classList.add('hidden');
                    collapseLeftBtn.classList.remove('hidden');
                    
                    if (map) {
                      setTimeout(() => { map.resize(); }, 300);
                    }
                  }
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
                
                // Show memento in live feed
                displayMementoInLiveFeed(memento);
              });
              
              // Add to markers array
              window.markers.push(marker);
          }
        } catch (error) {
          console.error('Error creating marker for memento:', error);
        }
      });
    } catch (error) {
      console.error('Error loading user mementos:', error);
        throw error;
    }
    });
  }

  // Function to load and display all user mementos on the map
  async function loadAllUserMementosOnMap() {
    try {
      console.log('Starting to load all user mementos...');
      showToast('Loading other users\' mementos...', 'info');
      
      // Get current user from Firebase auth
      const user = firebase.auth().currentUser;
      
      // If user is not logged in, just return
      if (!user) {
        console.warn('No authenticated user found when loading all user mementos');
        return;
      }

      console.log('Fetching all user mementos for user:', user.uid);
      // Get all mementos from Firestore
      const mementosSnapshot = await firebase.firestore()
        .collection('mementos')
        .where('userId', '!=', user.uid)
        .get();

      console.log('Retrieved all user mementos:', mementosSnapshot.size);
      let loadedCount = 0;
      mementosSnapshot.forEach((doc) => {
        try {
          const memento = { id: doc.id, ...doc.data() };
          
          // Only create marker if memento has location
          if (memento.location && memento.location.coordinates) {
            // Create marker element
            const el = document.createElement('div');
            el.className = 'all-user-memento-marker';
            
            // Set data-marker-type attribute for filtering
            el.setAttribute('data-marker-type', 'all-user-memento');
            
            // Set background image if available
            if (memento.media && memento.media.length > 0) {
              const firstMedia = memento.media[0];
              const mediaUrl = typeof firstMedia === 'string' ? firstMedia : firstMedia.url;
              const mediaType = typeof firstMedia === 'string' ? 'image' : firstMedia.type;
              
              if (mediaType === 'video' || mediaUrl.endsWith('.mp4') || mediaUrl.endsWith('.webm')) {
                // For videos, create a video element
                const video = document.createElement('video');
                video.src = mediaUrl;
                video.muted = true;
                video.loop = true;
                video.playsInline = true;
                video.style.width = '100%';
                video.style.height = '100%';
                video.style.objectFit = 'cover';
                el.appendChild(video);
                
                // Start playing when marker is added
                video.play().catch(error => {
                  console.log('Video autoplay failed:', error);
                });
              } else {
                // For images, use background image
                el.style.backgroundImage = `url(${mediaUrl})`;
              }
            }
            
            // Create marker
            const marker = new mapboxgl.Marker({
              element: el,
              anchor: 'bottom'
            })
              .setLngLat([memento.location.coordinates.longitude, memento.location.coordinates.latitude])
              .addTo(map);

            // Add click event to show memento details
            marker.getElement().addEventListener('click', () => {
              // Only expand panel on mobile devices
              if (window.innerWidth < 1024) {
                const infoTab = document.querySelector('.info-tab');
                const expandLeftBtn = document.getElementById('expand-left');
                const collapseLeftBtn = document.getElementById('collapse-left');
                
                if (infoTab && expandLeftBtn && collapseLeftBtn) {
                  infoTab.style.visibility = 'visible';
                  infoTab.classList.remove('hidden');
                  expandLeftBtn.classList.add('hidden');
                  collapseLeftBtn.classList.remove('hidden');
                  
                  if (map) {
                    setTimeout(() => { map.resize(); }, 300);
                  }
                }
              }
              
              displayMementoInLiveFeed(memento);
            });

            // Add to markers array
            window.markers.push(marker);
            loadedCount++;
            
            // Debug log
            console.log('Added all-user-memento marker:', {
              id: memento.id,
              type: el.getAttribute('data-marker-type'),
              location: memento.location.coordinates
            });
          }
        } catch (error) {
          console.error('Error creating marker for memento:', error);
        }
      });
      
      console.log(`Loaded ${loadedCount} all user mementos`);
      showToast(`Loaded ${loadedCount} other users' mementos`, 'success');
    } catch (error) {
      console.error('Error loading all user mementos:', error);
      showToast('Error loading other users\' mementos', 'error');
    }
  }

  // ---------------------------
  // 19) Update markers based on radius
  // ---------------------------
  function updateMarkersRadius() {
    if (!map) {
      console.log('Map not initialized');
      return;
    }

    const radiusToggle = document.getElementById('radius-toggle');
    const radiusSlider = document.getElementById('radius-slider');
    
    if (!radiusToggle || !radiusSlider) {
      console.log('Radius controls not found');
      return;
    }
    
    const isRadiusEnabled = radiusToggle.checked;
    const radiusMiles = parseFloat(radiusSlider.value);
    const center = userLocation || map.getCenter().toArray();
    const currentZoom = map.getZoom();

    if (!window.markers || !window.markers.length) {
      console.log('No markers to update');
      return;
    }

    // Batch DOM operations
    const updates = window.markers.map(marker => {
      if (!marker || !marker.getElement) return null;
      
      const markerElement = marker.getElement();
      if (!markerElement) return null;

      const coordinates = marker.getLngLat().toArray();
      const isInRadius = !isRadiusEnabled || isPointWithinRadius(coordinates, center, radiusMiles);
      
      return { element: markerElement, inRadius: isInRadius };
    }).filter(Boolean);

    // Apply all DOM updates in a single batch
    requestAnimationFrame(() => {
      updates.forEach(update => {
        const element = update.element;
        
        // Handle radius visibility
        if (update.inRadius) {
          element.classList.remove('out-of-radius');
          element.style.removeProperty('opacity');
          element.style.removeProperty('pointer-events');
          element.style.removeProperty('filter');
        } else {
          element.classList.add('out-of-radius');
          // Let CSS handle the styling for out-of-radius markers
        }

        // Update size based on zoom level
        element.classList.remove('zoom-level-0', 'zoom-level-1', 'zoom-level-2', 'zoom-level-3', 'zoom-level-4');
        let zoomLevel;
        if (currentZoom <= 12) {
          zoomLevel = 0;
        } else if (currentZoom <= 13) {
          zoomLevel = 1;
        } else if (currentZoom <= 14) {
          zoomLevel = 2;
        } else if (currentZoom <= 15) {
          zoomLevel = 3;
        } else {
          zoomLevel = 4;
        }
        element.classList.add(`zoom-level-${zoomLevel}`);
      });
    });
  }

  // Update radius slider event listener
  if (radiusSlider && radiusValue) {
    const debouncedUpdate = debounce((value) => {
      if (radiusToggle && radiusToggle.checked) {
        if (userLocation) {
          updateRadiusCircle(userLocation, parseFloat(value));
        } else if (map) {
          const center = map.getCenter().toArray();
          updateRadiusCircle(center, parseFloat(value));
      }
      }
      updateMarkersRadius();
      updateMap();
    }, 100); // 100ms debounce

    radiusSlider.addEventListener("input", function () {
      const formattedValue = Number(this.value).toFixed(2);
      radiusValue.innerText = formattedValue + ' mi';
      debouncedUpdate(formattedValue);
    });
  }

// ---------------------------
// 29) UI helper functions
// ---------------------------
  // Add this function to handle toast notifications
  function showToast(message, type = 'success') {
    // Check if splash screen is still visible
    const splashScreen = document.getElementById('splash-screen');
    if (splashScreen && !splashScreen.classList.contains('splash-hidden')) {
      // Delay toast until after splash screen is hidden
      setTimeout(() => showToast(message, type), 3500);
      return;
    }
    
    // Remove any existing toasts
    document.querySelectorAll('.toast').forEach(toast => toast.remove());
    
    // Create new toast
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    // Add icon based on type
    const icon = document.createElement('i');
    switch(type) {
      case 'success':
        icon.className = 'fas fa-check-circle';
        break;
      case 'error':
        icon.className = 'fas fa-exclamation-circle';
        break;
      case 'info':
        icon.className = 'fas fa-info-circle';
        break;
      case 'warning':
        icon.className = 'fas fa-exclamation-triangle';
        break;
    }
    
    toast.insertBefore(icon, toast.firstChild);
    document.body.appendChild(toast);
    
    // Trigger animation
    setTimeout(() => toast.classList.add('show'), 100);
    
    // Remove after delay
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  }




// ---------------------------
// 37) Explorer tab navigation
// ---------------------------  
  // Add this to your existing tab switching logic
  function switchExplorerTab(tabName) {
    // Hide all content sections
    const contentSections = document.querySelectorAll('.explorer-content');
    contentSections.forEach(section => {
      section.style.display = 'none';
    });

    // Deactivate all tab buttons
    const tabButtons = document.querySelectorAll('.explorer-tab');
    tabButtons.forEach(button => {
      button.classList.remove('active');
    });

    // Show selected content and activate its tab
    const selectedContent = document.getElementById(`${tabName}-content`);
    const selectedTab = document.querySelector(`.explorer-tab[data-tab="${tabName}"]`);
    
    if (selectedContent) {
      selectedContent.style.display = 'block';
    }
    
    if (selectedTab) {
      selectedTab.classList.add('active');
    }

    // If switching to drafts tab, show drafts container
    if (tabName === 'drafts') {
      const draftsContainer = document.getElementById('drafts-container');
      if (draftsContainer) {
        draftsContainer.style.display = 'block';
        initializeDraftsContainer();
      }
    }
  }

// ---------------------------
// 38) Radius circle management
// ---------------------------
  // Function to safely update radius UI elements
  function updateRadiusUI(isVisible) {
    if (radiusControl) {
        if (isVisible) {
            radiusControl.classList.remove('hidden');
        } else {
            radiusControl.classList.add('hidden');
        }
    }
  }

  // Update the updateRadiusCircle function to use the safe UI update
  function updateRadiusCircle(center, radiusMiles) {
    if (!map) return;
    
    // Remove existing radius circle
    removeRadiusCircle();
    
    if (radiusToggle && radiusToggle.checked) {
        updateRadiusUI(true);
        // Create and add the radius circle
        const circleData = createGeoJSONCircle(center, radiusMiles);
        
        map.addSource("radius-circle", {
            type: "geojson",
            data: circleData
        });
        
        map.addLayer({
            id: "radius-circle-fill",
            type: "fill",
            source: "radius-circle",
            paint: {
                "fill-color": "rgb(255, 255, 255)",
                "fill-opacity": 0.35
            }
        });
        
        map.addLayer({
            id: "radius-circle-outline",
            type: "line",
            source: "radius-circle",
            paint: {
                "line-color": "rgb(255, 255, 255)",
                "line-width": 2,
                "line-opacity": 0.8,
                "line-dasharray": [2, 2]
            }
        });
    } else {
        updateRadiusUI(false);
        removeRadiusCircle();
    }
    
    // Update markers within radius
    updateMarkersRadius();
    
    // Refresh live feed content if the function exists
    if (typeof window.refreshLiveFeedContent === 'function') {
        window.refreshLiveFeedContent();
    }
  }

  // Function to update slider range based on selected range type
  function updateSliderRange(rangeType) {
    if (rangeType === 'short') {
        // Short range: 10ft to 1mi
        radiusSlider.min = feetToMiles(10).toString();  // 10 feet in miles
        radiusSlider.max = "1";  // 1 mile
        radiusSlider.step = ((1 - feetToMiles(10)) / 100).toString();  // 100 steps
        radiusSlider.value = "0.5";  // Set default to middle of range
        minValueSpan.textContent = "10 ft";
        maxValueSpan.textContent = "1 mi";
        radiusValue.textContent = "0.5 mi";
    } else {
        // Long range: 1mi to 5mi
        radiusSlider.min = "1";
        radiusSlider.max = "5";
        radiusSlider.step = "0.1";  // 40 steps
        radiusSlider.value = "3";  // Set default to middle of range
        minValueSpan.textContent = "1 mi";
        maxValueSpan.textContent = "5 mi";
        radiusValue.textContent = "3 mi";
    }
    
    // Update the map with new radius
    if (userLocation) {
        updateRadiusCircle(userLocation, parseFloat(radiusSlider.value));
    }
    updateMap();
    updateMarkersRadius();
  }

  // Update radius toggle event listener
  if (radiusToggle) {
    radiusToggle.addEventListener("change", function() {
      if (this.checked) {
        // Show the circle
        if (userLocation) {
          const radiusMiles = radiusSlider ? parseFloat(radiusSlider.value) : 1.0;
          updateRadiusCircle(userLocation, radiusMiles);
        } else if (map) {
          const center = map.getCenter().toArray();
          const radiusMiles = radiusSlider ? parseFloat(radiusSlider.value) : 1.0;
          updateRadiusCircle(center, radiusMiles);
        }
      } else {
        // Hide the circle
        removeRadiusCircle();
      }
      updateMap();
    });
  }

  // Update radius slider event listener
  if (radiusSlider && radiusValue) {
    radiusSlider.addEventListener("input", function () {
      const formattedValue = Number(this.value).toFixed(2);
      radiusValue.innerText = formattedValue + ' mi';
      
      // Only update the circle if the radius toggle is checked
      if (radiusToggle && radiusToggle.checked) {
        if (userLocation) {
          updateRadiusCircle(userLocation, parseFloat(formattedValue));
        } else if (map) {
          const center = map.getCenter().toArray();
          updateRadiusCircle(center, parseFloat(formattedValue));
        }
      }
      updateMap();
    });
  }

  // Remove the map moveend event listener that was updating the circle
  // as it was causing the circle to move away from the user's location
  // map.on('moveend', function() { ... });


// ---------------------------
// 27) Filter management
// ---------------------------
  // Initialize all filters
  function initializeFilters() {
    const filterGroups = document.querySelectorAll('.filter-group');
    
    filterGroups.forEach(group => {
      // Reset to collapsed state
      group.classList.add('collapsed');
      group.classList.remove('expanded');
      
      // Add click handler to the header
      const header = group.querySelector('h3');
      header.addEventListener('click', () => {
        // Toggle collapsed/expanded state
        group.classList.toggle('collapsed');
        group.classList.toggle('expanded');
      });
    });
  }

  // Helper function to set up a filter group with consistent behavior
  function setupFilterGroup(name, toggleElement, containerElement, extraContainer = null) {
    if (!toggleElement || !containerElement) return;
    
    const filterHeader = toggleElement.closest('h3');
    const headerContent = filterHeader.querySelector('.filter-header-content');
    
    // Default state - expanded
    let isExpanded = true;
    
    // Function to update the expanded/collapsed state
    function updateExpandedState() {
      containerElement.style.display = isExpanded ? 'flex' : 'none';
      if (extraContainer) {
        extraContainer.style.display = isExpanded ? 'block' : 'none';
      }
    }
    
    // Toggle button only controls filter state
    toggleElement.addEventListener('change', (e) => {
      // Only update the filter state
      applyFilters();
    });
    
    // Header content controls expand/collapse
    headerContent.addEventListener('click', (e) => {
      // Toggle expanded state
      isExpanded = !isExpanded;
      updateExpandedState();
    });
    
    // Initialize UI
    updateExpandedState();
  }

  // Function to set up radius filter events
  function setupRadiusFilterEvents(radiusSlider, radiusRangeToggle) {
    if (!radiusSlider || !radiusRangeToggle) return;
    
    const radiusFilterGroup = document.querySelector('.filter-group');
    const filterHeader = radiusRangeToggle.closest('h3');
    const headerContent = filterHeader.querySelector('.filter-header-content');
    
    // Default state - expanded
    let isExpanded = true;
    
    // Function to update the expanded/collapsed state
    function updateExpandedState() {
      const container = radiusFilterGroup.querySelector('.radius-interaction-container');
      container.style.display = isExpanded ? 'flex' : 'none';
    }
    
    // Toggle button only controls filter state
    radiusRangeToggle.addEventListener('change', (e) => {
      // Only update the filter state
      applyFilters();
    });
    
    // Header content controls expand/collapse
    headerContent.addEventListener('click', (e) => {
      // Toggle expanded state
      isExpanded = !isExpanded;
      updateExpandedState();
    });
    
    // Initialize UI
    updateExpandedState();
    
    // ... rest of the existing radius filter event setup code ...
  }

  // Setup creators-specific filter events
  function setupCreatorsFilterEvents() {
    const creatorsToggle = document.getElementById('creators-toggle');
    const creatorsContainer = document.querySelector('.creators-interaction-container');
    const myMementosToggle = document.getElementById('filter-my-mementos');
    const allUsersToggle = document.getElementById('filter-all-users');
    const publicToggle = document.getElementById('filter-public');
    
    if (!creatorsToggle || !creatorsContainer || !myMementosToggle || !allUsersToggle || !publicToggle) {
      console.error('Creator filter elements not found');
      return;
    }

    // Use the setupFilterGroup function to handle the toggle and chevron
    setupFilterGroup('creators', creatorsToggle, creatorsContainer);

    // Handle individual toggles
    [myMementosToggle, allUsersToggle, publicToggle].forEach(toggle => {
      toggle.addEventListener('change', function() {
        const isEnabled = creatorsToggle.checked;
        if (isEnabled) {
          const showMyMementos = myMementosToggle.checked;
          const showAllUsers = allUsersToggle.checked;
          const showPublic = publicToggle.checked;
          updateMarkersByCreator(showMyMementos, showAllUsers, showPublic);
        }
      });
    });
    
    // Initial update
    const showMyMementos = creatorsToggle.checked && myMementosToggle.checked;
    const showAllUsers = creatorsToggle.checked && allUsersToggle.checked;
    const showPublic = creatorsToggle.checked && publicToggle.checked;
    updateMarkersByCreator(showMyMementos, showAllUsers, showPublic);
  }

  // Setup category-specific filter events
  function setupCategoryFilterEvents() {
    const categoryToggle = document.getElementById('category-toggle');
    const categoryContainer = document.querySelector('.category-interaction-container');
    
    if (!categoryToggle || !categoryContainer) {
      console.error('Category filter elements not found');
      return;
    }

    // Use the setupFilterGroup function to handle the toggle and chevron
    setupFilterGroup('category', categoryToggle, categoryContainer);

    // Add event listeners to category checkboxes
    document.querySelectorAll('.category-option input[type="checkbox"]').forEach(checkbox => {
      checkbox.addEventListener('change', applyFilters);
    });
  }

// ---------------------------
// 39) Filter application
// ---------------------------
  function applyFilters() {
    // Get filter states
    const radiusToggle = document.getElementById('radius-toggle');
    const creatorToggle = document.getElementById('creators-toggle');
    
    // Get all markers
    const markers = window.markers || [];
    
    markers.forEach(marker => {
      const markerElement = marker.getElement();
      if (!markerElement) return;
      
      let shouldShow = true;
      
      // Apply creator filter
      if (creatorToggle && creatorToggle.checked) {
        const showMyMementos = document.getElementById('filter-my-mementos').checked;
        const showAllUsers = document.getElementById('filter-all-users').checked;
        const showPublic = document.getElementById('filter-public').checked;
        
        if (marker._mementoData) {
          const isMyMemento = marker._mementoData.userId === firebase.auth().currentUser?.uid;
          const isPublic = !marker._mementoData.userId;
          
          if (!((isMyMemento && showMyMementos) || 
                (!isMyMemento && !isPublic && showAllUsers) || 
                (isPublic && showPublic))) {
            shouldShow = false;
          }
        }
      }
      
      // Update marker visibility
      markerElement.style.display = shouldShow ? 'block' : 'none';
      
      // Update marker classes for styling
      if (creatorToggle) {
        markerElement.classList.toggle('creator-filtered', !shouldShow);
      }
    });
    
    // Refresh the live feed content if available
    if (typeof window.refreshLiveFeedContent === 'function') {
      window.refreshLiveFeedContent();
    }
  }

  async function saveFilters() {
    try {
      const user = firebase.auth().currentUser;
      if (!user) {
        console.log('No user logged in');
        return;
      }
      
      const radiusToggle = document.getElementById('radius-toggle');
      const creatorToggle = document.getElementById('creators-toggle');
      
      const filters = {
        radius: {
          enabled: radiusToggle.checked,
          value: document.getElementById('radius-slider').value,
          range: document.getElementById('radius-range-toggle').value
        },
        creators: {
          enabled: creatorToggle.checked,
          selected: []
        }
      };
      
      // Get selected creators
      document.querySelectorAll('.creator-option input[type="checkbox"]').forEach(checkbox => {
        if (checkbox.checked) {
          filters.creators.selected.push(checkbox.id.replace('filter-', ''));
        }
      });
      
      await db.collection('users').doc(user.uid).update({
        filters: filters
      });
      
      showToast('Filters saved successfully');
    } catch (error) {
      console.error('Error saving filters:', error);
      showToast('Error saving filters', 'error');
    }
  }

  function resetFilters() {
    // Reset radius filter
    document.getElementById('radius-toggle').checked = true;
    document.getElementById('radius-slider').value = 0.5;
    document.getElementById('radius-range-toggle').value = 'short';
    document.querySelector('.radius-interaction-container').style.display = 'flex';
    
    // Reset creator filters
    document.getElementById('creators-toggle').checked = true;
    document.querySelectorAll('.creator-option input[type="checkbox"]').forEach(checkbox => {
      checkbox.checked = true;
    });
    document.querySelector('.creators-interaction-container').style.display = 'flex';
    
    // Apply the reset filters
    applyFilters();
    
    // Show confirmation
    showToast('Filters reset to default');
  }

  async function loadUserFilters(user) {
    try {
      const userDoc = await db.collection('users').doc(user.uid).get();
      const filters = userDoc.data()?.filters;
      
      if (!filters) {
        console.log('No saved filters found');
        return;
      }
      
      // Apply radius filter settings
      if (filters.radius) {
        const radiusToggle = document.getElementById('radius-toggle');
        const radiusContainer = document.querySelector('.radius-interaction-container');
        const radiusSlider = document.getElementById('radius-slider');
        const radiusRangeToggle = document.getElementById('radius-range-toggle');
        
        if (radiusToggle) radiusToggle.checked = filters.radius.enabled;
        if (radiusContainer) {
          radiusContainer.style.display = filters.radius.enabled ? 'flex' : 'none';
        }
        if (radiusSlider && filters.radius.value) {
          radiusSlider.value = filters.radius.value;
        }
        if (radiusRangeToggle && filters.radius.range) {
          radiusRangeToggle.value = filters.radius.range;
        }
      }
      
      // Apply creator filter settings
      if (filters.creators) {
        const creatorToggle = document.getElementById('creators-toggle');
        const creatorContainer = document.querySelector('.creators-interaction-container');
        const creatorCheckboxes = document.querySelectorAll('.creator-option input[type="checkbox"]');
        
        if (creatorToggle) creatorToggle.checked = filters.creators.enabled;
        if (creatorContainer) {
          creatorContainer.style.display = filters.creators.enabled ? 'flex' : 'none';
        }
        
        // Reset all checkboxes first
        creatorCheckboxes.forEach(checkbox => {
          checkbox.checked = false;
        });
        
        // Check selected creators
        if (filters.creators.selected && filters.creators.selected.length > 0) {
          filters.creators.selected.forEach(creator => {
            const checkbox = document.getElementById(`filter-${creator}`);
            if (checkbox) checkbox.checked = true;
          });
        } else {
          // If no creators selected, check all by default
          creatorCheckboxes.forEach(checkbox => {
            checkbox.checked = true;
          });
        }
      }
      
      // Apply the loaded filters
      applyFilters();
      
    } catch (error) {
      console.error('Error loading user filters:', error);
    }
  }

  // Add this to your initialization code
  document.addEventListener('DOMContentLoaded', () => {
    // ... existing initialization code ...
    initializeFilters();
  });

  // Add this utility function to check if right panel has content
  function checkAndUpdateRightPanel() {
    const rightPanel = document.querySelector('.right-panel');
    const expandRightBtn = document.getElementById('expand-right');
    const collapseRightBtn = document.getElementById('collapse-right');
    
    if (!rightPanel || !expandRightBtn || !collapseRightBtn) return;

    // Check if the right panel has any container elements
    const hasContainer = rightPanel.querySelector('.category-list-container') ||
                        rightPanel.querySelector('.event-list-container') ||
                        rightPanel.querySelector('.favorites-container');

    // Check if the container has meaningful content (not empty state)
    const hasContent = hasContainer && !(
        rightPanel.querySelector('.category-list-empty') ||
        rightPanel.querySelector('.event-list-empty') ||
        rightPanel.querySelector('.no-favorites')
    );
    
    if (!hasContainer || !hasContent) {
        // Collapse the panel if empty or no container
        rightPanel.classList.add('hidden');
        expandRightBtn.classList.remove('hidden');
        collapseRightBtn.classList.add('hidden');
        
        // Resize map after panel collapse
        if (map) {
            setTimeout(() => { map.resize(); }, 300);
        }
    }
  }


// ---------------------------
// 24) Firebase activity loading
// ---------------------------
  async function loadUserActivitiesFromFirebase() {
    try {
      // Wait for Firebase to be initialized
      if (!firebase.apps.length) {
        console.log('Firebase not initialized yet');
        return;
      }

      const user = firebase.auth().currentUser;
      if (!user) {
        console.log('User not logged in, activities will be loaded when user logs in');
        return;
      }
      
      // Initialize Firebase collections if needed
      await initializeFirebase();
      
      // Get user activities from Firebase
      const activities = await window.Activities.getUserActivities(user.uid);
      
      if (!activities || activities.length === 0) {
        return;
      }
      
      // Center map on the latest activity if any
      if (activities.length > 0 && activities[0].location && activities[0].location.coordinates) {
        const latestCoords = [
          activities[0].location.coordinates.longitude,
          activities[0].location.coordinates.latitude
        ];
        map.flyTo({
          center: latestCoords,
          zoom: 14,
          essential: true
        });
      }
    } catch (error) {
      console.error('Error loading activities:', error);
    }
  }

  // Hook up the Firebase data loading with authentication events
  firebase.auth().onAuthStateChanged(user => {
    if (user) {
      // User is signed in
      loadUserActivitiesFromFirebase();
    }
  });

  // Initialize map and Firebase data
  initializeAndUpdateMap();

  // Hook up the Firebase data loading with authentication events
  firebase.auth().onAuthStateChanged(user => {
    if (user) {
      // User is signed in
      loadUserActivitiesFromFirebase();
    }
  });

  // Pin on map button (now uses the main map for location selection)
  const pinOnMapBtn = document.getElementById('pin-on-map-btn');
  if (pinOnMapBtn) {
    pinOnMapBtn.addEventListener('click', function() {
      const isActive = this.classList.toggle('active');
      
      // If deactivating pin mode
      if (!isActive) {
        // Remove crosshair cursor
        document.body.classList.remove('pin-mode');
        
        // Remove instructions overlay
        const overlay = document.getElementById('map-instructions-overlay');
        if (overlay) {
          overlay.remove();
        }
        
        // Remove click handler from map
        if (map && window.mapClickHandler) {
          map.off('click', window.mapClickHandler);
          delete window.mapClickHandler;
        }
        
        // Remove temporary marker if it exists
        if (window.tempLocationMarker) {
          window.tempLocationMarker.remove();
          delete window.tempLocationMarker;
        }
        
        return;
        }
        
        // If activating pin mode
        if (isActive) {
          // Set cursor to crosshair
          document.body.classList.add('pin-mode');
          
          // Create and add instructions overlay
          const overlay = document.createElement('div');
          overlay.id = 'map-instructions-overlay';
          overlay.className = 'map-instructions-overlay';
          overlay.innerHTML = `
            <div class="instructions-content">
              <i class="fas fa-map-marker-alt"></i>
              <p>Click anywhere on the map to place a pin</p>
            </div>
          `;
          document.getElementById('discovery').appendChild(overlay);
          
          // Show toast notification
          showToast('Click anywhere on the map to select a location', 'info', 5000);
      }
    });
  }

  // Add event listener for radius changes to update list view
  if (radiusToggle) {
    radiusToggle.addEventListener("change", function() {
      const listViewBtn = document.querySelector('[data-tab="list-view"]');
      if (listViewBtn && listViewBtn.classList.contains('active')) {
        handleSubTabAction('discovery', 'list-view');
      }
    });
  }

  if (radiusSlider) {
    radiusSlider.addEventListener("input", function() {
      const listViewBtn = document.querySelector('[data-tab="list-view"]');
      if (listViewBtn && listViewBtn.classList.contains('active')) {
        handleSubTabAction('discovery', 'list-view');
      }
    });
  }

  // Ensure the panel is checked when the page loads
  document.addEventListener('DOMContentLoaded', function() {
    // Initial check of right panel
    checkAndUpdateRightPanel();
  });

  // Helper function for capture form visibility
  function handleCaptureFormVisibility(activity, tabId) {
    const captureForm = document.getElementById('journey-capture-form');
    if (!captureForm) return;

    if (tabId === 'capture' && activity === 'journey') {
        captureForm.classList.remove('hidden');
        setTimeout(() => {
            captureForm.style.visibility = 'visible';
        }, 300);
    } else {
        captureForm.classList.add('hidden');
    }
  }

  // Initialize map with user location
  mapboxgl.accessToken = 'pk.eyJ1IjoiMDIwOXZhaWJoYXYiLCJhIjoiY2x6cW4xY2w5MWswZDJxcHhreHZ2OG5mbSJ9.ozamGsW5CZrZdL5bG7n_0A';
  
  // Request location before initializing map
  if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
          position => {
              userLocation = [position.coords.longitude, position.coords.latitude];
              // Wait for splash screen to complete before initializing map
              const splashScreen = document.querySelector('.splash-screen');
              if (splashScreen) {
                  splashScreen.addEventListener('animationend', () => {
                      // Initialize map centered on NYC after splash screen
                      initializeMap();
                      
                      // Then after 5 seconds, zoom to user location
                      setTimeout(() => {
                          map.flyTo({
                              center: userLocation,
                              zoom: 14,
                              duration: 2000,
                              essential: true
                          });
                          updateMap();
                      }, 5000); // Wait 5 seconds before zooming to location
                  });
              } else {
                  // If no splash screen, initialize immediately
                  initializeMap();
                  map.flyTo({
                      center: userLocation,
                      zoom: 14,
                      duration: 2000,
                      essential: true
                  });
                  updateMap();
              }
          },
          error => {
              console.error("Initial Position Error:", error);
              const splashScreen = document.querySelector('.splash-screen');
              if (splashScreen) {
                  splashScreen.addEventListener('animationend', () => {
                      initializeMap();
                      updateMap();
                  });
              } else {
                  initializeMap();
                  updateMap();
              }
          },
          { 
              enableHighAccuracy: true,
              timeout: 5000,
              maximumAge: 0
          }
      );
        } else {
      initializeMap();
      setTimeout(() => {
          initializeAndUpdateMap();
      }, 1000);
  }

  // Initialize UI state
  initializeUIState();

  function initializeUIState() {
    // Set Discovery as active tab
    document.querySelectorAll('.activity-tab').forEach(tab => tab.classList.remove('active'));
    const discoveryTab = document.querySelector('[data-activity="discovery"]');
    if (discoveryTab) {
      discoveryTab.classList.add('active');
      currentTab = 'discovery';
    }

    // Hide all subtabs
    document.querySelectorAll('.explorer-subtabs').forEach(subtab => {
      subtab.classList.remove('show');
    });

    // Check if device is mobile
    const isMobile = window.innerWidth <= 768;

    // Set left panel state based on device type
    if (isMobile) {
      // For mobile: collapsed by default
      infoTab.classList.add('hidden');
      collapseLeftBtn.classList.add('hidden');
      expandLeftBtn.classList.remove('hidden');
    } else {
      // For desktop: expanded by default
      infoTab.classList.remove('hidden');
      collapseLeftBtn.classList.remove('hidden');
      expandLeftBtn.classList.add('hidden');
    }

    // Resize map after panel state changes
    setTimeout(() => {
      if (map) {
        map.resize();
      }
    }, 300);
  }

  // Function to load public memento markers from JSON
  async function loadPublicMementoMarkers() {
    try {
      console.log('Loading public memento markers...');
      
      // Clear existing markers
      if (window.markers) {
        window.markers.forEach(marker => marker.remove());
      }
      window.markers = [];

      // Load public mementos data
      const publicMementos = await loadPublicMementosData();
      
      // Create markers for each public memento
      publicMementos.forEach(memento => {
        try {
          // Create marker element
          const el = document.createElement('div');
          el.className = 'public-memento-marker';
          
          // Set data-marker-type attribute for filtering
          el.setAttribute('data-marker-type', 'public-memento');
          
          // Set background image if available
          if (memento.media && memento.media.length > 0) {
            const firstMedia = memento.media[0];
            const mediaUrl = typeof firstMedia === 'string' ? firstMedia : firstMedia.url;
            const mediaType = typeof firstMedia === 'string' ? 'image' : firstMedia.type;
            
            if (mediaType === 'video' || mediaUrl.endsWith('.mp4') || mediaUrl.endsWith('.webm')) {
              // For videos, create a video element
              const video = document.createElement('video');
              video.src = mediaUrl;
              video.muted = true;
              video.loop = true;
              video.playsInline = true;
              video.style.width = '100%';
              video.style.height = '100%';
              video.style.objectFit = 'cover';
              el.appendChild(video);
              
              // Start playing when marker is added
              video.play().catch(error => {
                console.log('Video autoplay failed:', error);
              });
            } else {
              // For images, use background image
              el.style.backgroundImage = `url(${mediaUrl})`;
            }
          }
          
          // Create marker
          const marker = new mapboxgl.Marker({
            element: el,
            anchor: 'bottom'
          })
            .setLngLat([memento.location.longitude, memento.location.latitude])
            .addTo(map);

          // Store memento ID with marker
          marker.mementoId = memento.id;
          
          // Add click event to show memento details
          marker.getElement().addEventListener('click', () => {
            // Only expand panel on mobile devices
            if (window.innerWidth < 1024) {
              const infoTab = document.querySelector('.info-tab');
              const expandLeftBtn = document.getElementById('expand-left');
              const collapseLeftBtn = document.getElementById('collapse-left');
              
              if (infoTab && expandLeftBtn && collapseLeftBtn) {
                infoTab.style.visibility = 'visible';
                infoTab.classList.remove('hidden');
                expandLeftBtn.classList.add('hidden');
                collapseLeftBtn.classList.remove('hidden');
                
                if (map) {
                  setTimeout(() => { map.resize(); }, 300);
                }
              }
            }
            
            displayMementoInLiveFeed(memento);
          });
          
          // Add marker to global markers array
          window.markers.push(marker);
        } catch (error) {
          console.error('Error creating marker:', error);
        }
      });

      // Update markers radius after creating them
      updateMarkersRadius();
    } catch (error) {
      console.error('Error loading public memento markers:', error);
    }
  }

  // Update markers based on radius
  function updateMarkersRadius() {
    if (!map) {
      console.log('Map not initialized');
      return;
    }

    const radiusToggle = document.getElementById('radius-toggle');
    const radiusSlider = document.getElementById('radius-slider');
              
    if (!radiusToggle || !radiusSlider) {
      console.log('Radius controls not found');
      return;
    }
    
    const isRadiusEnabled = radiusToggle.checked;
    const radiusMiles = parseFloat(radiusSlider.value);
    const center = userLocation || map.getCenter().toArray();
    const currentZoom = map.getZoom();

    if (!window.markers || !window.markers.length) {
      console.log('No markers to update');
      return;
    }

    // Batch DOM operations
    const updates = window.markers.map(marker => {
      if (!marker || !marker.getElement) return null;
      
      const markerElement = marker.getElement();
      if (!markerElement) return null;

      const coordinates = marker.getLngLat().toArray();
      const isInRadius = !isRadiusEnabled || isPointWithinRadius(coordinates, center, radiusMiles);
      
      return { element: markerElement, inRadius: isInRadius };
    }).filter(Boolean);

    // Apply all DOM updates in a single batch
    requestAnimationFrame(() => {
      updates.forEach(update => {
        const element = update.element;
        
        // Handle radius visibility
        if (update.inRadius) {
          element.classList.remove('out-of-radius');
          element.style.removeProperty('opacity');
          element.style.removeProperty('pointer-events');
          element.style.removeProperty('filter');
        } else {
          element.classList.add('out-of-radius');
          // Let CSS handle the styling for out-of-radius markers
        }

        // Update size based on zoom level
        element.classList.remove('zoom-level-0', 'zoom-level-1', 'zoom-level-2', 'zoom-level-3', 'zoom-level-4');
        let zoomLevel;
        if (currentZoom <= 12) {
          zoomLevel = 0;
        } else if (currentZoom <= 13) {
          zoomLevel = 1;
        } else if (currentZoom <= 14) {
          zoomLevel = 2;
        } else if (currentZoom <= 15) {
          zoomLevel = 3;
        } else {
          zoomLevel = 4;
        }
        element.classList.add(`zoom-level-${zoomLevel}`);
      });
    });
  }

  // Add styles for public markers and popups
  const publicStyle = document.createElement('style');
  publicStyle.textContent = `
    .public-memento-popup, .user-memento-popup {
      max-width: 300px;
    }

    .public-memento-popup-content, .user-memento-popup-content {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .public-memento-popup-media, .user-memento-popup-media {
      width: 100%;
      height: 150px;
      overflow: hidden;
      border-radius: 8px;
    }

    .public-memento-popup-media img, .user-memento-popup-media img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }

    .public-memento-details, .user-memento-details {
      padding: 10px;
    }

    .public-memento-details h3, .user-memento-details h3 {
      margin: 0 0 5px 0;
      font-size: 16px;
      color: #111827;
    }

    .public-memento-details p, .user-memento-details p {
      margin: 0 0 5px 0;
      font-size: 14px;
      color: #4b5563;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .public-memento-details i, .user-memento-details i {
      color: #FF0080;
    }

    .public-memento-tags, .user-memento-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      margin-top: 10px;
    }

    .public-memento-tags .tag, .user-memento-tags .tag {
      background-color: #FF0080;
      color: white;
      padding: 4px 8px;
      border-radius: 12px;
      font-size: 12px;
    }
  `;
  document.head.appendChild(publicStyle);

  // ... existing code ...

  // Function to scroll to live feed
  function scrollToLiveFeed() {
    const liveFeedContainer = document.querySelector('.live-feed-container');
    if (liveFeedContainer) {
      liveFeedContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  // Unified function to display memento info
  window.displayMementoInfo = async function(memento) {
    try {
          // Get username if available
          let username = 'Anonymous';
      if (memento.userId) {
          if (memento.mementoType === 'public') {
          // For public mementos, use the userId directly as the source/author
          username = memento.userId;
        } else if (!memento.userId.includes('/') && !memento.userId.includes(' ')) {
            // Only query Firebase if userId looks like a Firebase ID (no spaces or slashes)
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
        } else {
            // If userId exists but contains spaces or slashes, use it directly
            username = memento.userId;
        }
      }

      // Format date and time
      const timestamp = new Date(memento.timestamp);
      const formattedDateTime = timestamp.toLocaleString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: 'numeric',
        minute: 'numeric'
      });

      // Create media HTML
      let mediaHtml = `
        <div class="placeholder-media">
          <i class="fas fa-eye"></i>
        </div>
      `;

      if (memento.media && memento.media.length > 0) {
            const firstMedia = memento.media[0];
            if (typeof firstMedia === 'string') {
              // Check if it's a video by extension
              if (firstMedia.endsWith('.mp4') || firstMedia.endsWith('.webm')) {
                mediaHtml = `<video src="${firstMedia}" controls playsinline></video>`;
              } else {
                mediaHtml = `<img src="${firstMedia}" alt="${memento.name || 'Memento'}" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\\'placeholder-media\\'><i class=\\'fas fa-eye\\'></i></div>';">`;
              }
            } else if (firstMedia && typeof firstMedia === 'object') {
              const mediaUrl = firstMedia.url || firstMedia.path || firstMedia.src;
              if (mediaUrl) {
                if (firstMedia.type === 'video' || mediaUrl.endsWith('.mp4') || mediaUrl.endsWith('.webm')) {
                  mediaHtml = `<video src="${mediaUrl}" controls playsinline></video>`;
                } else {
                  mediaHtml = `<img src="${mediaUrl}" alt="${memento.name || 'Memento'}" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\\'placeholder-media\\'><i class=\\'fas fa-eye\\'></i></div>';">`;
                }
              }
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

              ${memento.mementoType === 'public' ? `
                ${memento.timestamp ? `
                    <p class="memento-timestamp">
                      <i class="fas fa-clock"></i>
                      ${memento.timestamp}
                    </p>
                ` : ''}

                ${memento.mementoDuration ? `
                  <p class="memento-duration">
                    <i class="fas fa-hourglass-half"></i>
                    ${memento.mementoDuration}
                  </p>
                ` : ''}
              ` : `
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
              `}

              ${memento.userId ? `
                    <p class="memento-author">
                      <i class="fas fa-user"></i>
                      ${username}
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
                    ${memento.mementoType === 'public' ? `
                ${memento.category ? `
                  <div class="memento-category">
                    <span class="category-symbol">${memento.category.match(/^[\u{1F300}-\u{1F9FF}]/u)?.[0] || ''}</span>
                    <span class="category-name">${memento.category.replace(/^[\u{1F300}-\u{1F9FF}]/u, '').trim()}</span>
                  </div>
                ` : ''}

                ${memento.mementoTags && memento.mementoTags.length > 0 ? `
                  <div class="memento-tags">
                    ${memento.mementoTags.slice(0, 3).map(tag => `
                        <span class="tag">
                        <span class="tag-symbol">${tag.match(/^[\u{1F300}-\u{1F9FF}]/u)?.[0] || ''}</span>
                        <span class="tag-name">${tag.replace(/^[\u{1F300}-\u{1F9FF}]/u, '').trim()}</span>
                        </span>
                    `).join('')}
                    ${memento.mementoTags.length > 3 ? `
                        <span class="tag">
                        <span class="tag-symbol">...</span>
                        <span class="tag-name">+${memento.mementoTags.length - 3} more</span>
                        </span>
                    ` : ''}
                  </div>
                ` : ''}
              ` : `
                ${memento.category ? `
                  <div class="memento-category">
                    <span class="category-symbol">${getCategorySymbol(memento.category)}</span>
                    <span class="category-name">${window.categories.find(c => c.id === memento.category)?.name || ''}</span>
                  </div>
                ` : ''}

                ${memento.tags && memento.tags.length > 0 ? `
                  <div class="memento-tags">
                    ${memento.tags.slice(0, 3).map(tag => {
                      const tagInfo = window.tags.find(t => t.id === tag) || { symbol: '', name: '' };
                        return `<span class="tag">
                          <span class="tag-symbol">${tagInfo.symbol}</span>
                          <span class="tag-name">${tagInfo.name}</span>
                        </span>`;
                    }).join('')}
                    ${memento.tags.length > 3 ? `
                        <span class="tag">
                        <span class="tag-symbol">...</span>
                        <span class="tag-name">+${memento.tags.length - 3} more</span>
                        </span>
                    ` : ''}
                  </div>
                ` : ''}
              `}
                </div>
              </div>
            </div>
          `;

      // Switch to memento-info tab and show content
      const mementoInfoTab = document.querySelector('.live-feed-tab-btn[data-tab="memento-info"]');
      const mementoInfoContent = document.getElementById('memento-info-content');
      const mementoInfoList = mementoInfoContent.querySelector('.memento-info-list');

      if (mementoInfoTab) {
        mementoInfoTab.click();
      }

      if (mementoInfoContent) {
        mementoInfoContent.classList.add('active');
        mementoInfoContent.style.display = 'block';
      }

          // Update the memento info list content
      if (mementoInfoList) {
          mementoInfoList.innerHTML = mementoInfoHtml;

          // Add click handler for the view-on-map button
          const viewOnMapBtn = mementoInfoList.querySelector('.view-on-map-btn');
          if (viewOnMapBtn && memento.location && memento.location.coordinates) {
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
                  zoom: 15,
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

        // Add click handler for the like button
        const likeBtn = mementoInfoList.querySelector('.like-btn');
        if (likeBtn) {
          likeBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            try {
              const mementoId = likeBtn.dataset.mementoId;
              const currentUser = firebase.auth().currentUser;
              
              if (!currentUser) {
                showToast('Please sign in to like mementos', 'warning');
                return;
              }

              // Get the memento document reference
              const mementoRef = firebase.firestore().collection('mementos').doc(mementoId);
              const mementoDoc = await mementoRef.get();
              
              if (!mementoDoc.exists) {
                showToast('Memento not found', 'error');
                return;
              }

              const mementoData = mementoDoc.data();
              const likes = mementoData.likes || [];
              const isLiked = likes.includes(currentUser.uid);

              if (isLiked) {
                // Unlike
                await mementoRef.update({
                  likes: firebase.firestore.FieldValue.arrayRemove(currentUser.uid),
                  likeCount: firebase.firestore.FieldValue.increment(-1)
                });
                likeBtn.classList.remove('liked');
              } else {
                // Like
                await mementoRef.update({
                  likes: firebase.firestore.FieldValue.arrayUnion(currentUser.uid),
                  likeCount: firebase.firestore.FieldValue.increment(1)
                });
                likeBtn.classList.add('liked');
              }

              // Update like count display
              const likeCount = likeBtn.querySelector('.like-count');
              if (likeCount) {
                const currentCount = parseInt(likeCount.textContent);
                likeCount.textContent = isLiked ? currentCount - 1 : currentCount + 1;
              }

              showToast(isLiked ? 'Memento unliked' : 'Memento liked', 'success');
            } catch (error) {
              console.error('Error handling like:', error);
              showToast('Error updating like status', 'error');
            }
          });
        }

        // Add click handler for the favorite button
        const favoriteBtn = mementoInfoList.querySelector('.favorite-btn');
        if (favoriteBtn) {
          favoriteBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            try {
              const mementoId = favoriteBtn.dataset.mementoId;
              const currentUser = firebase.auth().currentUser;
              
              if (!currentUser) {
                showToast('Please sign in to favorite mementos', 'warning');
                return;
              }

              // Get the memento document reference
              const mementoRef = firebase.firestore().collection('mementos').doc(mementoId);
              const mementoDoc = await mementoRef.get();
              
              if (!mementoDoc.exists) {
                showToast('Memento not found', 'error');
                return;
              }

              const mementoData = mementoDoc.data();
              const favorites = mementoData.favorites || [];
              const isFavorited = favorites.includes(currentUser.uid);

              if (isFavorited) {
                // Unfavorite
                await mementoRef.update({
                  favorites: firebase.firestore.FieldValue.arrayRemove(currentUser.uid),
                  favoriteCount: firebase.firestore.FieldValue.increment(-1)
                });
                favoriteBtn.classList.remove('favorited');
              } else {
                // Favorite
                await mementoRef.update({
                  favorites: firebase.firestore.FieldValue.arrayUnion(currentUser.uid),
                  favoriteCount: firebase.firestore.FieldValue.increment(1)
                });
                favoriteBtn.classList.add('favorited');
              }

              // Update favorite count display
              const favoriteCount = favoriteBtn.querySelector('.favorite-count');
              if (favoriteCount) {
                const currentCount = parseInt(favoriteCount.textContent);
                favoriteCount.textContent = isFavorited ? currentCount - 1 : currentCount + 1;
              }

              showToast(isFavorited ? 'Memento removed from favorites' : 'Memento added to favorites', 'success');
            } catch (error) {
              console.error('Error handling favorite:', error);
              showToast('Error updating favorite status', 'error');
            }
          });
        }
      }
    } catch (error) {
      console.error('Error displaying memento info:', error);
      showToast('Error displaying memento details. Please try again.', 'error');
    }
  }

  // Update the existing displayMementoInLiveFeed function to use the new unified function
  async function displayMementoInLiveFeed(memento) {
    try {
      // Show live feed container
      const liveFeedContainer = document.querySelector('.live-feed-container');
      if (liveFeedContainer) {
        liveFeedContainer.classList.remove('hidden');
      }
      
      // Use the unified display function
      await displayMementoInfo(memento);
    } catch (error) {
      console.error('Error displaying memento in live feed:', error);
      showToast('Error displaying memento details. Please try again.', 'error');
    }
  }

  // Function to edit a memento
  window.editMemento = async function(memento) {
    try {
      // Ensure categories, durations, and tags are loaded
      if (!window.categories || !window.durations || !window.tags) {
        await loadCategoriesAndTags();
      }

      // Switch to journey tab and show capture form
      const journeyTab = document.querySelector('.activity-tab[data-activity="journey"]');
      if (journeyTab) {
        // Remove active class from all tabs
        document.querySelectorAll('.activity-tab').forEach(tab => tab.classList.remove('active'));
        // Add active class to journey tab
        journeyTab.classList.add('active');
        currentTab = 'journey';

        // Hide all subtabs
        document.querySelectorAll('.explorer-subtabs').forEach(subtab => {
          subtab.classList.remove('show');
        });

        // Show journey subtabs
        const journeySubtabs = journeyTab.nextElementSibling;
        if (journeySubtabs && journeySubtabs.classList.contains('explorer-subtabs')) {
          journeySubtabs.classList.add('show');
        }

        // Activate capture subtab
        const captureTab = journeySubtabs.querySelector('[data-tab="capture"]');
        if (captureTab) {
          journeySubtabs.querySelectorAll('.explorer-tab-btn').forEach(btn => btn.classList.remove('active'));
          captureTab.classList.add('active');
          handleSubTabAction('journey', 'capture');
        }
      }

      // Show the journey capture form
      const captureForm = document.getElementById('journey-capture-form');
      if (!captureForm) {
        throw new Error('Capture form not found');
      }
      captureForm.classList.remove('hidden');

      // Populate form fields with memento data
      const nameInput = document.getElementById('memento-name');
      const descriptionInput = document.getElementById('memento-description');
      const locationInput = document.getElementById('memento-location');
      const timestampInput = document.getElementById('memento-timestamp');
      const mediaPreview = document.getElementById('media-preview');
      const mediaPreviewText = document.getElementById('media-preview-text');

      // Set name
      if (nameInput) nameInput.value = memento.name || '';

      // Set description
      if (descriptionInput) descriptionInput.value = memento.description || '';

      // Set location
      if (locationInput && memento.location) {
        locationInput.value = memento.location.address || '';
      }

      // Set timestamp
      if (timestampInput && memento.timestamp) {
        const date = memento.timestamp.toDate ? memento.timestamp.toDate() : new Date(memento.timestamp);
        timestampInput.value = formatDateTimeForInput(date);
      }

      // Clear existing selections
      const selectedTagsContainer = document.getElementById('selected-tags');
      const selectedCategoriesContainer = document.getElementById('selected-categories');
      const selectedDurationsContainer = document.getElementById('selected-durations');
      
      if (selectedTagsContainer) selectedTagsContainer.innerHTML = '';
      if (selectedCategoriesContainer) selectedCategoriesContainer.innerHTML = '';
      if (selectedDurationsContainer) selectedDurationsContainer.innerHTML = '';

      // Set category
      if (memento.category && window.categories) {
        const category = window.categories.find(c => c.id === memento.category);
        if (category) {
          window.addSelectedItem('category', category.id, category.name, category.symbol);
        }
      }

      // Set duration
      if (memento.duration && window.durations) {
        const duration = window.durations.find(d => d.id === memento.duration);
        if (duration) {
          window.addSelectedItem('duration', duration.id, duration.name, duration.symbol);
        }
      }

      // Set tags
      if (memento.tags && memento.tags.length > 0 && window.tags) {
        memento.tags.forEach(tagId => {
          const tag = window.tags.find(t => t.id === tagId);
          if (tag) {
            window.addSelectedItem('tag', tag.id, tag.name, tag.symbol);
          }
        });
      }

      // Set media
      if (mediaPreview && memento.media && memento.media.length > 0) {
        mediaPreview.innerHTML = '';
        memento.media.forEach(media => {
          const mediaUrl = typeof media === 'string' ? media : media.url;
          if (mediaUrl) {
            const previewItem = document.createElement('div');
            previewItem.className = 'media-preview-item';
            if (mediaUrl.endsWith('.mp4') || mediaUrl.endsWith('.webm')) {
              const video = document.createElement('video');
              video.src = mediaUrl;
              video.controls = true;
              previewItem.appendChild(video);
            } else {
              const img = document.createElement('img');
              img.src = mediaUrl;
              img.alt = 'Memento media';
              previewItem.appendChild(img);
            }
            mediaPreview.appendChild(previewItem);
          }
        });
        if (mediaPreviewText) {
          mediaPreviewText.textContent = `${memento.media.length} media items`;
        }
      }

      // Change save button text to indicate edit mode
      const saveBtn = document.getElementById('save-memento-btn');
      if (saveBtn) {
        saveBtn.textContent = 'Update Memento';
      }

      // Show the first step
      showStep(1);

    } catch (error) {
      console.error('Error editing memento:', error);
      showToast('Error loading memento for editing. Please try again.', 'error');
    }
  };

  // Function to set up tags filter events
  function setupTagsFilterEvents() {
    const tagsToggle = document.getElementById('tags-toggle');
    const tagsContainer = document.querySelector('.filter-group:has(#tags-toggle) .category-interaction-container');
    const filterHeader = tagsToggle.closest('h3');
    const chevronIcon = filterHeader.querySelector('.fa-chevron-down');
    const chevronContainer = filterHeader.querySelector('.category-toggle-container');
    
    // Default state - expanded
    let isExpanded = true;
    
    // Function to update the expanded/collapsed state
    function updateExpandedState() {
      tagsContainer.style.display = isExpanded ? 'flex' : 'none';
      // Update chevron rotation
      if (chevronIcon) {
        chevronIcon.style.transform = isExpanded ? 'rotate(0deg)' : 'rotate(180deg)';
      }
    }
    
    // Toggle button controls filter on/off and visibility
    tagsToggle.addEventListener('change', () => {
      // When toggling on, expand the filter
      if (tagsToggle.checked) {
        isExpanded = true;
        updateExpandedState();
      } else {
        // When toggling off, collapse the filter
        isExpanded = false;
        updateExpandedState();
      }
      applyFilters();
    });

    // Chevron icon controls expanded/collapsed state
    chevronContainer.addEventListener('click', (e) => {
      // Only handle clicks on the container or chevron, not the toggle button
      if (e.target === tagsToggle) return;
      
      e.preventDefault();
      e.stopPropagation();
      
      // Toggle expanded state
      isExpanded = !isExpanded;
      updateExpandedState();
    });
    
    // Initialize UI
    updateExpandedState();
  }

  // Live Feed Tabs Functionality
  function initializeLiveFeedTabs() {
    // This function is now implemented in live-feed.js
    return window.initializeLiveFeedTabs();
  }

  async function loadCategoriesContent() {
    // This function is now implemented in live-feed.js
    return window.loadCategoriesContent();
  }

  async function loadTagsContent() {
    // This function is now implemented in live-feed.js
    return window.loadTagsContent();
  }

  async function loadDurationContent() {
    // This function is now implemented in live-feed.js
    return window.loadDurationContent();
  }

  async function getAllMementos() {
    try {
      const db = firebase.firestore();
      const mementosSnapshot = await db.collection('mementos').get();
      const mementos = mementosSnapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
      
      // Get current map center and radius
      const map = window.map;
      if (!map) return mementos;
      
      const center = map.getCenter();
      const radiusMiles = parseFloat(document.getElementById('radius-slider').value);
      
      // Filter mementos within radius
      return mementos.filter(memento => {
        if (!memento.location || !memento.location.coordinates) return false;
        const point = {
          type: 'Point',
          coordinates: memento.location.coordinates
        };
        return isPointWithinRadius(point, center, radiusMiles);
      });
    } catch (error) {
      console.error('Error fetching mementos:', error);
      return [];
    }
  }

  // Initialize live feed tabs when the page loads
  document.addEventListener('DOMContentLoaded', () => {
    if (typeof window.initializeLiveFeedTabs === 'function') {
      window.initializeLiveFeedTabs();
    }
  });

  function updateMarkerSizes(zoom) {
    // Remove all zoom level classes first
    document.querySelectorAll('.user-memento-marker, .public-memento-marker, .all-user-memento-marker').forEach(marker => {
      marker.classList.remove('zoom-level-0', 'zoom-level-1', 'zoom-level-2', 'zoom-level-3', 'zoom-level-4');
    });

    // Add appropriate zoom level class based on current zoom
    let zoomLevel;
    if (zoom <= 12) {
      zoomLevel = 0;
    } else if (zoom <= 13) {
      zoomLevel = 1;
    } else if (zoom <= 14) {
      zoomLevel = 2;
    } else if (zoom <= 15) {
      zoomLevel = 3;
    } else {
      zoomLevel = 4;
    }

    document.querySelectorAll('.user-memento-marker, .public-memento-marker, .all-user-memento-marker').forEach(marker => {
      marker.classList.add(`zoom-level-${zoomLevel}`);
    });
  }
}); // End of DOMContentLoaded event listener

// Update radius UI based on visibility
function updateRadiusUI(isVisible) {
  const radiusSlider = document.getElementById('radius-slider');
  const radiusRangeToggle = document.getElementById('radius-range-toggle');
  const radiusValue = document.getElementById('radius-value');
  
  if (isVisible) {
    radiusSlider.disabled = false;
    radiusRangeToggle.disabled = false;
    radiusValue.style.opacity = '1';
  } else {
    radiusSlider.disabled = true;
    radiusRangeToggle.disabled = true;
    radiusValue.style.opacity = '0.5';
  }
}

// Add this function to handle filter container toggle
function initializeFilterToggle() {
  const collapsedContainer = document.querySelector('.collapsed-filters-container');
  const filterSettingsIcon = document.querySelector('.filter-settings-icon');

  if (filterSettingsIcon && collapsedContainer) {
    // Set initial state
    let isCollapsed = collapsedContainer.classList.contains('collapsed');
    
    filterSettingsIcon.addEventListener('click', () => {
      isCollapsed = !isCollapsed;
      collapsedContainer.classList.toggle('collapsed', isCollapsed);
    });
  }
}

// Call this function when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  // ... existing initialization code ...
  initializeFilterToggle();
});

function setupRadiusFilterEvents(radiusSlider, radiusRangeToggle) {
    if (!radiusSlider || !radiusRangeToggle) return;
    
    const radiusFilterGroup = document.querySelector('.filter-group');
    const radiusToggle = document.getElementById('radius-toggle');
    const radiusContainer = document.querySelector('.radius-interaction-container');
    const sliderContainer = document.querySelector('.slider-container');
    const filterHeader = radiusRangeToggle.closest('h3');
    const headerContent = filterHeader.querySelector('.filter-header-content');
    const chevronIcon = filterHeader.querySelector('.fa-chevron-down');
    
    // Default state - expanded
    let isExpanded = true;
    
    // Function to update the expanded/collapsed state
    function updateExpandedState() {
      radiusContainer.style.display = isExpanded ? 'flex' : 'none';
      sliderContainer.style.display = isExpanded ? 'block' : 'none';
      // Update chevron rotation
      if (chevronIcon) {
        chevronIcon.style.transform = isExpanded ? 'rotate(0deg)' : 'rotate(180deg)';
      }
    }
    
    // Helper function to get center coordinates in correct format
    function getCenterCoordinates() {
      const center = map.getCenter();
      return [center.lng, center.lat];
    }
    
    // Helper function to format radius value
    function formatRadiusValue(value) {
      return Number(value).toFixed(1);
    }
    
    // Toggle button only controls filter state
    radiusToggle.addEventListener('change', () => {
      // Only control the GeoJSON circle visibility
      if (radiusToggle.checked) {
        // Show the circle
        if (map && map.getCenter()) {
          const radiusMiles = parseFloat(radiusSlider.value);
          updateRadiusCircle(getCenterCoordinates(), radiusMiles);
          updateMarkersRadius();
        }
      } else {
        // Hide the circle
        if (window.radiusCircle) {
          map.removeLayer(window.radiusCircle);
          window.radiusCircle = null;
        }
        // Reset all markers to be visible
        if (window.markers) {
          window.markers.forEach(marker => {
            marker.getElement().style.opacity = '1';
            marker.getElement().style.pointerEvents = 'auto';
          });
        }
      }
      applyFilters();
    });
    
    // Header content controls expand/collapse
    headerContent.addEventListener('click', (e) => {
      // Toggle expanded state
      isExpanded = !isExpanded;
      updateExpandedState();
    });
    
    // Range toggle changes the slider range
    radiusRangeToggle.addEventListener('change', () => {
      updateSliderRange(radiusRangeToggle.value);
      // Update the circle if the filter is enabled
      if (radiusToggle.checked && map && map.getCenter()) {
        updateRadiusCircle(getCenterCoordinates(), parseFloat(radiusSlider.value));
        updateMarkersRadius();
      }
    });
    
    // Slider changes update the radius circle
    radiusSlider.addEventListener('input', () => {
      const radiusValue = document.getElementById('radius-value');
      radiusValue.textContent = `${formatRadiusValue(radiusSlider.value)} mi`;
      
      if (radiusToggle.checked && map && map.getCenter()) {
        updateRadiusCircle(getCenterCoordinates(), parseFloat(radiusSlider.value));
        updateMarkersRadius();
      }
    });
    
    // Initialize UI
    updateExpandedState();
    
    // Initialize the radius circle if the toggle is checked
    if (radiusToggle.checked && map && map.getCenter()) {
      updateRadiusCircle(getCenterCoordinates(), parseFloat(radiusSlider.value));
      updateMarkersRadius();
    }
  }

  // ... existing code ...

  // Function to ensure Firebase is initialized
  async function ensureFirebaseInitialized() {
    try {
      // Check if Firebase is already initialized
      if (firebase.apps.length === 0) {
        console.log('Firebase not initialized, initializing...');
        await initializeFirebase();
      }
      
      // Check if user is authenticated
      const currentUser = firebase.auth().currentUser;
      if (!currentUser) {
        console.log('No authenticated user, waiting for auth state...');
        return new Promise((resolve, reject) => {
          const unsubscribe = firebase.auth().onAuthStateChanged(user => {
            unsubscribe();
            if (user) {
              console.log('User authenticated:', user.uid);
              resolve(user);
            } else {
              console.warn('User not authenticated after waiting');
              reject(new Error('User not authenticated'));
            }
          });
        });
      }
      
      return currentUser;
    } catch (error) {
      console.error('Error ensuring Firebase initialization:', error);
      throw error;
    }
  }

  async function loadMementosWithRetry(loadFunction, maxRetries = 3) {
    let retries = 0;
    while (retries < maxRetries) {
      try {
        await loadFunction();
        return; // Success, exit the function
      } catch (error) {
        retries++;
        console.warn(`Attempt ${retries} failed:`, error);
        
        if (retries === maxRetries) {
          console.error('Max retries reached, giving up');
          throw error;
        }
        
        // Wait before retrying (exponential backoff)
        const delay = Math.pow(2, retries) * 1000;
        console.log(`Retrying in ${delay}ms...`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }

  // Update the getCategorySymbol function to use window.categories
  function getCategorySymbol(categoryId) {
    if (!window.categories) return '🗂️';
    const category = window.categories.find(c => c.id === categoryId);
    return category ? category.symbol : '🗂️';
  }

  // Update the getTagSymbol function to use window.tags
  function getTagSymbol(tagId) {
    if (!window.tags) return '🏷️';
    const tag = window.tags.find(t => t.id === tagId);
    return tag ? tag.symbol : '🏷️';
  }

  // Function to create heatmap data from mementos
  function createHeatmapData(mementos) {
    return mementos.map(memento => ({
      type: 'Feature',
      properties: {
        intensity: 1
      },
      geometry: {
        type: 'Point',
        coordinates: [
          memento.location.coordinates.longitude,
          memento.location.coordinates.latitude
        ]
      }
    }));
  }

  // Function to toggle between markers and heatmap
  function toggleMarkersAndHeatmap(zoom) {
    const mapContainer = document.getElementById('map');
    const shouldShowHeatmap = zoom <= 13;

    if (shouldShowHeatmap) {
      // Show heatmap, hide markers
      if (map.getSource('mementos')) {
        map.setLayoutProperty('mementos-heatmap', 'visibility', 'visible');
      }
      mapContainer.classList.add('heatmap-visible');
      isHeatmapVisible = true;
      // Update heatmap data to show count labels
      updateHeatmapData();
    } else {
      // Hide heatmap, show markers
      if (map.getSource('mementos')) {
        map.setLayoutProperty('mementos-heatmap', 'visibility', 'none');
      }
      mapContainer.classList.remove('heatmap-visible');
      isHeatmapVisible = false;
      // Remove all count labels
      const existingLabels = document.querySelectorAll('.cluster-count-label');
      existingLabels.forEach(label => label.remove());
    }
  }

  // Function to update heatmap data
  function updateHeatmapData() {
    if (!map || !map.getSource('mementos')) return;

    const allMementos = [
      ...userMementoMarkers.map(m => m._lngLat),
      ...publicMementoMarkers.map(m => m._lngLat),
      ...allUserMementoMarkers.map(m => m._lngLat)
    ];

    // Create a grid to count mementos in each cell
    const gridSize = 0.005; // roughly 500m at equator
    const clusters = new Map();

    // Group mementos into grid cells
    allMementos.forEach(coords => {
      const gridX = Math.floor(coords.lng / gridSize);
      const gridY = Math.floor(coords.lat / gridSize);
      const key = `${gridX},${gridY}`;
      
      if (!clusters.has(key)) {
        clusters.set(key, {
          count: 0,
          lngSum: 0,
          latSum: 0
        });
      }
      
      const cluster = clusters.get(key);
      cluster.count++;
      cluster.lngSum += coords.lng;
      cluster.latSum += coords.lat;
    });

    // Convert clusters to features and find max count
    let maxCount = 0;
    const features = [];

    clusters.forEach((cluster, key) => {
      if (cluster.count > maxCount) {
        maxCount = cluster.count;
      }

      // Calculate center of cluster
      const centerLng = cluster.lngSum / cluster.count;
      const centerLat = cluster.latSum / cluster.count;

      features.push({
        type: 'Feature',
        properties: {
          intensity: cluster.count,
          count: cluster.count
        },
        geometry: {
          type: 'Point',
          coordinates: [centerLng, centerLat]
        }
      });
    });

    // Add count labels for significant clusters
    const threshold = 2; // Show counts for all clusters with 2 or more mementos
    const currentZoom = map.getZoom();
    
    // Remove existing count labels
    const existingLabels = document.querySelectorAll('.cluster-count-label');
    existingLabels.forEach(label => label.remove());

    // Only show count labels if zoom level is 13 or less
    if (currentZoom <= 13) {
      // Add new count labels for significant clusters
      features.forEach(feature => {
        if (feature.properties.count >= threshold) {
          const [lng, lat] = feature.geometry.coordinates;
          const point = map.project([lng, lat]);
          
          const label = document.createElement('div');
          label.className = 'cluster-count-label';
          label.style.left = `${point.x}px`;
          label.style.top = `${point.y}px`;
          label.textContent = feature.properties.count;
          label.dataset.lng = lng;
          label.dataset.lat = lat;
          
          document.getElementById('map').appendChild(label);
        }
      });
    }

    const heatmapData = {
      type: 'FeatureCollection',
      features: features
    };

    map.getSource('mementos').setData(heatmapData);
  }

  // ... existing code ...

  // Initialize map with user location
  mapboxgl.accessToken = 'pk.eyJ1IjoiMDIwOXZhaWJoYXYiLCJhIjoiY2x6cW4xY2w5MWswZDJxcHhreHZ2OG5mbSJ9.ozamGsW5CZrZdL5bG7n_0A';

  // Initialize map
  function initializeMap(center = [-73.935242, 40.730610]) { // Center of NYC
    map = new mapboxgl.Map({
      container: 'map',
      style: 'mapbox://styles/mapbox/light-v11',
      center: center,
      zoom: 12,
      attributionControl: false
    });

    // Add navigation controls
    map.addControl(new mapboxgl.NavigationControl(), 'top-right');
    
    // Add locate me control
    map.addControl(new LocateMeControl());

    // Add scale control
    map.addControl(new mapboxgl.ScaleControl({
      maxWidth: 100,
      unit: 'imperial'
    }), 'bottom-right');

    map.on('load', () => {
      // Initialize radius circle by default
      const radiusToggle = document.getElementById('radius-toggle');
      const radiusSlider = document.getElementById('radius-slider');
      
      if (radiusToggle) {
        radiusToggle.checked = true;
      }
      
      if (radiusSlider) {
        const radiusMiles = parseFloat(radiusSlider.value);
        const center = userLocation || map.getCenter().toArray();
        updateRadiusCircle(center, radiusMiles);
        updateMarkersRadius();
      }
    });

    // Handle map style changes
    map.on('style.load', () => {
      // Initialize any style-dependent features here
    });

    return map;
  }

  // Update markers based on radius
  function updateMarkersRadius() {
    if (!map) {
      console.log('Map not initialized');
      return;
    }

    const radiusToggle = document.getElementById('radius-toggle');
    const radiusSlider = document.getElementById('radius-slider');
    
    if (!radiusToggle || !radiusSlider) {
      console.log('Radius controls not found');
      return;
    }
    
    const isRadiusEnabled = radiusToggle.checked;
    const radiusMiles = parseFloat(radiusSlider.value);
    const center = userLocation || map.getCenter().toArray();
    const currentZoom = map.getZoom();

    if (!window.markers || !window.markers.length) {
      console.log('No markers to update');
      return;
    }

    // Batch DOM operations
    const updates = window.markers.map(marker => {
      if (!marker || !marker.getElement) return null;
      
      const markerElement = marker.getElement();
      if (!markerElement) return null;

      const coordinates = marker.getLngLat().toArray();
      const isInRadius = !isRadiusEnabled || isPointWithinRadius(coordinates, center, radiusMiles);
      
      console.log('Updating marker:', {
        coordinates,
        isInRadius,
        currentClass: markerElement.className
      });

      markerElement.classList.toggle('out-of-radius', !isInRadius);
      return { element: markerElement, visible: isInRadius };
    }).filter(Boolean);

    // Apply all DOM updates in a single batch
    requestAnimationFrame(() => {
      updates.forEach(update => {
        update.element.style.opacity = update.visible ? '1' : '0.3';
        update.element.style.pointerEvents = update.visible ? 'auto' : 'none';
      });
    });
  }

  // Function to add selected items (tags, categories, durations)
  window.addSelectedItem = function(type, value, text, symbol) {
    const container = document.getElementById(`selected-${type}s`);
    if (!container) return;

    // Check if item is already selected
    const existingItem = container.querySelector(`[data-value="${value}"]`);
    if (existingItem) return;

    // Create new selected item
    const item = document.createElement('div');
    item.className = 'selected-item';
    item.dataset.value = value;
    item.innerHTML = `
      <span class="item-symbol">${symbol}</span>
      <span class="item-name">${text}</span>
      <button class="remove-item" type="button">
        <i class="fas fa-times"></i>
      </button>
    `;

    // Add remove button functionality
    const removeBtn = item.querySelector('.remove-item');
    if (removeBtn) {
      removeBtn.addEventListener('click', () => {
        item.remove();
        // Re-enable the option in the select
        const select = document.getElementById(`${type}-select`);
        if (select) {
          const option = select.querySelector(`option[value="${value}"]`);
          if (option) {
            option.disabled = false;
          }
        }
      });
    }

    // Add to container
    container.appendChild(item);

    // Disable the option in the select
    const select = document.getElementById(`${type}-select`);
    if (select) {
      const option = select.querySelector(`option[value="${value}"]`);
      if (option) {
        option.disabled = true;
      }
    }
  };

  // Function to show a specific step in the journey capture form
  window.showStep = function(stepNumber) {
    // Hide all steps
    document.querySelectorAll('.form-step').forEach(step => {
      step.classList.remove('active');
    });

    // Show the selected step
    const stepToShow = document.querySelector(`.form-step[data-step="${stepNumber}"]`);
    if (stepToShow) {
      stepToShow.classList.add('active');
    }

    // Update progress bar
    const progressBar = document.querySelector('.progress-bar');
    if (progressBar) {
      const progress = ((stepNumber - 1) / 4) * 100; // 4 steps total
      progressBar.style.width = `${progress}%`;
    }

    // Update step indicators
    document.querySelectorAll('.step-indicator').forEach((indicator, index) => {
      if (index + 1 < stepNumber) {
        indicator.classList.add('completed');
      } else if (index + 1 === stepNumber) {
        indicator.classList.add('active');
      } else {
        indicator.classList.remove('active', 'completed');
      }
    });
  };

