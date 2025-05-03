// ---------------------------
// 1) Firebase initialization
// ---------------------------
// Initialize Firebase Auth
const auth = firebase.auth();
const db = firebase.firestore();
const storage = firebase.storage();

// ---------------------------
// 2) Auth state management
// ---------------------------
// Auth state observer
auth.onAuthStateChanged((user) => {
  const profileContent = document.getElementById('profile-tab');
  if (user) {
    // User is signed in
    showUserProfile(user);
  } else {
    // No user is signed in
    showAuthForms();
  }
});

// ---------------------------
// 3) Authentication UI
// ---------------------------
function showAuthForms() {
  let authContainer = document.getElementById('auth-container');
  if (!authContainer) {
    authContainer = document.createElement('div');
    authContainer.id = 'auth-container';
    authContainer.className = 'auth-container';
    
    // Insert the auth container below subtabs
    const subtabs = document.querySelector('.explorer-subtabs');
    if (subtabs) {
      subtabs.insertAdjacentElement('afterend', authContainer);
    } else {
      // If subtabs don't exist yet, append to main content
      const mainContent = document.querySelector('.main-content');
      if (mainContent) {
        mainContent.appendChild(authContainer);
      }
    }
  }
  
  authContainer.classList.remove('hidden');
  authContainer.innerHTML = `
    <div class="auth-tabs">
      <button class="auth-tab active" data-auth="login">Login</button>
      <button class="auth-tab" data-auth="signup">Sign Up</button>
    </div>
    
    <form id="loginForm" class="auth-form">
      <div class="form-group">
        <label for="loginEmail">Email</label>
        <input type="email" id="loginEmail" value="" required>
      </div>
      <div class="form-group">
        <label for="loginPassword">Password</label>
        <input type="password" id="loginPassword" value="" required>
      </div>
      <button type="submit" class="auth-button">Login</button>
    </form>

    <form id="signupForm" class="auth-form hidden">
      <div class="form-group">
        <label for="signupName">Full Name</label>
        <input type="text" id="signupName" value="" required>
      </div>
      <div class="form-group">
        <label for="signupUsername">Username</label>
        <div class="input-wrapper">
          <span class="at-symbol">@</span>
          <input type="text" id="signupUsername" value="" pattern="[a-zA-Z0-9]{3,20}" required>
        </div>
        <p class="field-note">Username must be 3-20 characters and can only contain letters and numbers</p>
      </div>
      <div class="form-group">
        <label for="signupEmail">Email</label>
        <input type="email" id="signupEmail" value="" required>
      </div>
      <div class="form-group">
        <label for="signupPassword">Password</label>
        <input type="password" id="signupPassword" value="" required>
      </div>
      <button type="submit" class="auth-button">Sign Up</button>
    </form>
  `;

  setupAuthFormListeners();
}

// ---------------------------
// 4) User data management
// ---------------------------
// Function to get latest user data from Firebase
async function refreshUserData(user) {
  if (!user) {
    console.error('No user provided to refreshUserData');
    return null;
  }

  try {
    const userRef = db.collection('users').doc(user.uid);
    const doc = await userRef.get();
    
    if (!doc.exists) {
      // If document doesn't exist, create it with basic user info
      const basicUserData = {
        displayName: user.displayName || '',
        email: user.email || '',
        username: user.email.split('@')[0], // Add default username
        phoneNumber: '',
        addresses: [],
        bio: '',
        interests: [],
        photoURL: user.photoURL || 'default-avatar.png',
        createdAt: firebase.firestore.FieldValue.serverTimestamp(),
        updatedAt: firebase.firestore.FieldValue.serverTimestamp()
      };

      try {
        await userRef.set(basicUserData);
        console.log('Created new user document:', user.uid);
        return basicUserData;
      } catch (createError) {
        console.error('Error creating user document:', createError);
        throw new Error('Failed to create user profile');
      }
    }
    
    return doc.data();
  } catch (error) {
    console.error('Error in refreshUserData:', error);
    throw error;
  }
}

// ---------------------------
// 5) Profile completion metrics
// ---------------------------
// Add this function to calculate profile completion
function calculateProfileCompletion(userData) {
  const fields = {
    displayName: { weight: 15, validate: v => v && v.length > 0 },
    photoURL: { weight: 10, validate: v => v && v !== 'default-avatar.png' },
    phoneNumber: { weight: 15, validate: v => v && v.length === 10 },
    addresses: { weight: 20, validate: v => v && Array.isArray(v) && v.length > 0 },
    bio: { weight: 15, validate: v => v && v.length > 0 },
    interests: { weight: 15, validate: v => v && Array.isArray(v) && v.length > 0 }
  };

  let completionScore = 0;
  for (const [field, { weight, validate }] of Object.entries(fields)) {
    if (validate(userData[field])) {
      completionScore += weight;
    }
  }

  return completionScore;
}

// ---------------------------
// 6) Profile display UI
// ---------------------------
// Update showUserProfile to include completion indicator
async function showUserProfile(user) {
  if (!user) {
    console.error('No user provided to showUserProfile');
    showAuthForms();
    return;
  }

  let authContainer = document.getElementById('auth-container');
  if (!authContainer) {
    authContainer = document.createElement('div');
    authContainer.id = 'auth-container';
    authContainer.className = 'auth-container';
    
    // Insert the auth container below subtabs
    const subtabs = document.querySelector('.explorer-subtabs');
    if (subtabs) {
      subtabs.insertAdjacentElement('afterend', authContainer);
    } else {
      // If subtabs don't exist yet, append to main content
      const mainContent = document.querySelector('.main-content');
      if (mainContent) {
        mainContent.appendChild(authContainer);
      }
    }
  }

  try {
    const userData = await refreshUserData(user);
    if (!userData) {
      throw new Error('Could not load user data');
    }
    
    const completionScore = calculateProfileCompletion(userData);
    
    authContainer.innerHTML = `
      <div class="profile-container">
        <div class="profile-completion">
          <div class="completion-bar">
            <div class="completion-fill" style="width: ${completionScore}%"></div>
          </div>
          <span class="completion-text">Profile ${completionScore}% Complete</span>
        </div>

        <div class="detail-section">
          <div class="detail-row" data-type="photo">
            <div class="profile-photo-container">
                <img src="${userData.photoURL || 'default-avatar.png'}" 
                     alt="Profile" 
                     class="profile-photo"
                     onerror="this.onerror=null; this.src='default-avatar.png';">
                <div class="photo-upload-overlay">
                    <label for="photoUpload" class="upload-label">
                        <i class="fas fa-camera"></i>
                        <span>Change Photo</span>
                    </label>
                    <input type="file" id="photoUpload" accept="image/*" style="display: none;">
                </div>
            </div>
          </div>

          <div class="detail-row" data-type="username">
            <span>Username</span>
            <span class="detail-value">@${userData.username || userData.email.split('@')[0]}</span>
            <button class="chevron-btn"></button>
          </div>

          <div class="detail-row" data-type="name">
            <span>Name</span>
            <span class="detail-value">${userData.displayName || 'Add name'}</span>
            <button class="chevron-btn"></button>
          </div>

          <div class="detail-row" data-type="email">
            <span>Email</span>
            <span class="detail-value">${userData.email}</span>
            <button class="chevron-btn"></button>
          </div>

          <div class="detail-row" data-type="phone">
            <span>Phone</span>
            <span class="detail-value">${userData.phoneNumber || 'Add phone number'}</span>
            <button class="chevron-btn"></button>
          </div>

          <div class="detail-row" data-type="address">
            <span>Address</span>
            <span class="detail-value">${formatAddressPreview(userData.addresses)}</span>
            <button class="chevron-btn"></button>
          </div>

          <div class="detail-row" data-type="bio">
            <span>Bio</span>
            <span class="detail-value">${userData.bio || 'Add bio'}</span>
            <button class="chevron-btn"></button>
          </div>

          <div class="detail-row" data-type="interests">
            <span>Interests</span>
            <span class="detail-value">${userData.interests ? userData.interests.join(', ') : 'Add interests'}</span>
            <button class="chevron-btn"></button>
          </div>

          <div class="detail-row" data-type="interested-categories">
            <span>Interested Categories</span>
            <span class="detail-value">${userData.interestedCategories ? userData.interestedCategories.map(id => getCategorySymbol(id)).join(' ') : 'Add categories'}</span>
            <button class="chevron-btn"></button>
          </div>

          <div class="detail-row" data-type="interested-tags">
            <span>Interested Tags</span>
            <span class="detail-value">${userData.interestedTags ? userData.interestedTags.map(id => getTagSymbol(id)).join(' ') : 'Add tags'}</span>
            <button class="chevron-btn"></button>
          </div>
        </div>

        <div class="profile-actions">
          <button id="logoutBtn" class="profile-action-btn">Log out</button>
        </div>
      </div>
    `;

    setupProfileEventListeners(user);
  } catch (error) {
    console.error('Error in showUserProfile:', error);
    showAuthForms();
  }
}

// ---------------------------
// 7) Helper functions & formatting
// ---------------------------
// Update the formatAddressPreview function to handle undefined fields
function formatAddressPreview(addresses) {
  if (!addresses || !Array.isArray(addresses) || addresses.length === 0) {
    return 'Add address';
  }
  
  const addr = addresses[0];
  if (!addr) return 'Add address';

  let preview = addr.street || '';
  if (addr.apartment) preview += `, ${addr.apartment}`;
  if (addr.city) preview += `, ${addr.city}`;
  if (addr.state) preview += `, ${addr.state}`;
  if (addr.zipCode) preview += ` ${addr.zipCode}`;
  
  return preview || 'Add address';
}

// ---------------------------
// 8) Event listeners & interactions
// ---------------------------
function setupProfileEventListeners(user) {
  const items = document.querySelectorAll('.detail-row');
  
  items.forEach(item => {
    const type = item.getAttribute('data-type');
    const editBtn = item.querySelector('.chevron-btn');
    
    if (editBtn) {
      editBtn.addEventListener('click', () => {
        switch(type) {
          case 'photo':
            handleImageUpload(user);
            break;
          case 'username':
            showEditUsernameScreen(user);
            break;
          case 'name':
            showEditNameScreen(user);
            break;
          case 'phone':
            showEditPhoneScreen(user);
            break;
          case 'address':
            showEditAddressScreen(user);
            break;
          case 'bio':
            showEditBioScreen(user);
            break;
          case 'interests':
            showEditInterestsScreen(user);
            break;
          case 'interested-categories':
            showEditInterestedCategoriesScreen(user);
            break;
          case 'interested-tags':
            showEditInterestedTagsScreen(user);
            break;
        }
      });
    }
  });

  // Edit Profile button
  const editProfileBtn = document.getElementById('editProfileBtn');
  if (editProfileBtn) {
    editProfileBtn.addEventListener('click', () => {
      showEditProfileModal(user);
    });
  }

  // Logout button
  const logoutBtn = document.getElementById('logoutBtn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', () => {
      auth.signOut();
    });
  }

  setupPhotoUploadListener(user);
}

// ---------------------------
// 9) UI element utilities
// ---------------------------
// Helper function to find elements by text content
function querySelector(selector) {
  return Array.from(document.querySelectorAll(selector))
    .find(el => el.textContent.trim() === text);
}

function togglePhoneEdit(isEditing, field, editBtn) {
  const span = field.querySelector('span');
  const input = field.querySelector('input');
  const actions = field.querySelector('.edit-actions');

  span.classList.toggle('hidden', isEditing);
  input.classList.toggle('hidden', !isEditing);
  actions.classList.toggle('hidden', !isEditing);
  editBtn.classList.toggle('hidden', isEditing);
}

function setupAvatarEdit(user) {
  const editAvatarBtn = document.querySelector('.edit-avatar-btn');
  if (editAvatarBtn) {
    editAvatarBtn.addEventListener('click', () => {
      handleImageUpload(user);
    });
  }
}

async function handleImageUpload(user) {
  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = 'image/*';
  fileInput.click();

  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
      try {
        // Validate file size (5MB max)
        if (file.size > 5 * 1024 * 1024) {
          throw new Error('File size must be less than 5MB');
        }

        // Show loading state
        showLoadingState('avatar');

        // Create storage reference
        const storageRef = storage.ref();
        const fileRef = storageRef.child(`users/${user.uid}/avatar-${Date.now()}`);
        
        // Upload file with metadata
        const metadata = {
          contentType: file.type,
          customMetadata: {
            'uploadedBy': user.uid,
            'uploadedAt': new Date().toISOString()
          }
        };
        
        // Upload file
        const uploadTask = fileRef.put(file, metadata);

        // Monitor upload progress (optional)
        uploadTask.on('state_changed', 
          (snapshot) => {
            const progress = (snapshot.bytesTransferred / snapshot.totalBytes) * 100;
            console.log('Upload progress: ' + progress + '%');
          },
          (error) => {
            console.error('Upload error:', error);
            hideLoadingState('avatar');
            alert('Failed to upload image');
          },
          async () => {
            // Upload completed successfully
            const downloadURL = await fileRef.getDownloadURL();

            // Update profile and database
            await user.updateProfile({ photoURL: downloadURL });
            await db.collection('users').doc(user.uid).update({
              photoURL: downloadURL,
              updatedAt: firebase.firestore.FieldValue.serverTimestamp()
            });

            // Update UI
            document.querySelector('.profile-photo').src = downloadURL;
            hideLoadingState('avatar');
          }
        );

      } catch (error) {
        console.error('Error updating avatar:', error);
        alert(error.message || 'Failed to update avatar');
        hideLoadingState('avatar');
      }
    }
  });
}

function setupBioEdit(user) {
  const editBioBtn = document.querySelector('.edit-bio-btn');
  if (editBioBtn) {
    editBioBtn.addEventListener('click', () => {
      const bioText = document.querySelector('.profile-bio p').textContent;
      const bioContainer = document.querySelector('.profile-bio');
      
      bioContainer.innerHTML = `
        <textarea class="bio-editor" maxlength="200">${bioText}</textarea>
        <div class="bio-actions">
          <button class="save-bio-btn">Save</button>
          <button class="cancel-bio-btn">Cancel</button>
        </div>
      `;

      setupBioActionListeners(user, bioContainer, bioText);
    });
  }
}

function setupBioActionListeners(user, bioContainer, originalBioText) {
  document.querySelector('.save-bio-btn').addEventListener('click', async () => {
    const newBio = document.querySelector('.bio-editor').value.trim();
    try {
      await db.collection('users').doc(user.uid).update({
        bio: newBio,
        updatedAt: firebase.firestore.FieldValue.serverTimestamp()
      });
      updateBioDisplay(bioContainer, newBio);
      setupProfileEventListeners(user);
    } catch (error) {
      console.error('Error updating bio:', error);
      alert('Failed to update bio');
    }
  });

  document.querySelector('.cancel-bio-btn').addEventListener('click', () => {
    updateBioDisplay(bioContainer, originalBioText);
    setupProfileEventListeners(user);
  });
}

function setupProfileActions(user) {
  // Edit Profile
  const editProfileBtn = document.querySelector('.edit-profile');
  if (editProfileBtn) {
    editProfileBtn.addEventListener('click', () => {
      showEditProfileModal(user);
    });
  }

  // Change Password
  const changePasswordBtn = document.querySelector('.change-password');
  if (changePasswordBtn) {
    changePasswordBtn.addEventListener('click', () => {
      showChangePasswordModal(user);
    });
  }
}

// ---------------------------
// 10) Profile editing functions
// ---------------------------
function showEditProfileModal(user) {
  // Replace the entire profile content with the edit form
  const authContainer = document.getElementById('auth-container');
  authContainer.innerHTML = `
    <div class="profile-container">
      <div class="edit-screen-header">
        <button class="back-btn">‹</button>
        <h3>Edit Profile</h3>
      </div>
      
      <form id="editProfileForm" class="edit-profile-form">
        <div class="edit-field-group">
          <label for="displayName">Name</label>
          <input type="text" id="displayName" value="${user.displayName || ''}" required>
        </div>
        
        <div class="edit-field-group">
          <label for="email">Email</label>
          <input type="email" id="email" value="${user.email}" disabled>
          <p class="field-note">Email cannot be changed</p>
        </div>

        <div class="edit-field-group">
          <label for="phone">Phone Number</label>
          <input type="tel" id="phone" value="${user.phoneNumber || ''}" placeholder="(123) 456-7890">
        </div>

        <div class="edit-field-group">
          <label for="street">Address</label>
          <input type="text" id="street" value="${user.address?.street || ''}" placeholder="Enter your street address">
        </div>

        <div class="edit-field-group">
          <label for="apartment">Apartment/Suite (Optional)</label>
          <input type="text" id="apartment" value="${user.address?.apartment || ''}" placeholder="Apt, Suite, Unit, etc.">
        </div>

        <div class="edit-field-group">
          <label for="bio">Bio</label>
          <textarea id="bio" maxlength="200" placeholder="Tell us about yourself">${user.bio || ''}</textarea>
          <p class="field-note">Maximum 200 characters</p>
        </div>

        <div class="edit-field-group">
          <label for="interests">Interests (comma-separated)</label>
          <textarea id="interests" placeholder="e.g., photography, travel, music">${(user.interests || []).join(', ')}</textarea>
          <p class="field-note">Separate multiple interests with commas</p>
        </div>
        
        <div class="form-actions">
          <button type="button" class="cancel-btn">Cancel</button>
          <button type="submit" class="save-btn">Update Profile</button>
        </div>
      </form>
    </div>
  `;

  // Add back button and cancel button listeners
  const backToProfile = () => showUserProfile(user);
  document.querySelector('.back-btn').addEventListener('click', backToProfile);
  document.querySelector('.cancel-btn').addEventListener('click', backToProfile);

  // Add form submit listener
  document.getElementById('editProfileForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    try {
      const newData = {
        displayName: document.getElementById('displayName').value.trim(),
        phoneNumber: document.getElementById('phone').value.trim(),
        address: {
          street: document.getElementById('street').value.trim(),
          apartment: document.getElementById('apartment').value.trim()
        },
        bio: document.getElementById('bio').value.trim(),
        interests: document.getElementById('interests').value
          .split(',')
          .map(interest => interest.trim())
          .filter(interest => interest.length > 0),
        updatedAt: firebase.firestore.FieldValue.serverTimestamp()
      };

      // Update display name in Firebase Auth
      await user.updateProfile({ displayName: newData.displayName });
      
      // Update user data in Firestore
      await db.collection('users').doc(user.uid).update(newData);
      
      // Show success message
      showToast('Profile updated successfully');
      
      // Return to profile view
      showUserProfile(user);
    } catch (error) {
      console.error('Error updating profile:', error);
      showToast('Failed to update profile', 'error');
    }
  });
}

// ---------------------------
// 11) Password change functions
// ---------------------------
function showChangePasswordModal(user) {
  const modalHTML = `
    <div class="modal-content">
      <h3>Change Password</h3>
      <form id="changePasswordForm">
        <div class="form-group">
          <label for="currentPassword">Current Password</label>
          <input type="password" id="currentPassword" required>
        </div>
        <div class="form-group">
          <label for="newPassword">New Password</label>
          <input type="password" id="newPassword" required>
        </div>
        <div class="form-group">
          <label for="confirmPassword">Confirm New Password</label>
          <input type="password" id="confirmPassword" required>
        </div>
        <div class="modal-actions">
          <button type="button" class="cancel-btn">Cancel</button>
          <button type="submit" class="save-btn">Change Password</button>
        </div>
      </form>
    </div>
  `;

  showModal(modalHTML, async () => {
    const form = document.getElementById('changePasswordForm');
    const newPassword = form.newPassword.value;
    const confirmPassword = form.confirmPassword.value;

    if (newPassword !== confirmPassword) {
      alert('New passwords do not match');
      return;
    }

    try {
      await user.updatePassword(newPassword);
      hideModal();
      alert('Password updated successfully');
    } catch (error) {
      console.error('Error updating password:', error);
      alert('Failed to update password');
    }
  });
}

// Helper functions
function showLoadingState(type) {
  const photoContainer = document.querySelector(`[data-type="${type}"]`);
  if (photoContainer) {
    const btn = photoContainer.querySelector('.chevron-btn');
    if (btn) {
      btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    }
  }
}

function hideLoadingState(type) {
  const photoContainer = document.querySelector(`[data-type="${type}"]`);
  if (photoContainer) {
    const btn = photoContainer.querySelector('.chevron-btn');
    if (btn) {
      btn.innerHTML = '›';
    }
  }
}

function updateBioDisplay(container, text) {
  container.innerHTML = `
    <p>${text}</p>
    <button class="edit-bio-btn"><i class="fas fa-edit"></i></button>
  `;
}

// ---------------------------
// 12) Modal & notification utilities
// ---------------------------
function showModal(content, onSubmit) {
  const modal = document.createElement('div');
  modal.className = 'modal';
  modal.innerHTML = content;
  document.body.appendChild(modal);

  const form = modal.querySelector('form');
  const cancelBtn = modal.querySelector('.cancel-btn');

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    onSubmit();
  });

  cancelBtn.addEventListener('click', hideModal);
}

function hideModal() {
  const modal = document.querySelector('.modal');
  if (modal) {
    modal.remove();
  }
}

function setupLogout() {
  const logoutBtn = document.getElementById('logoutBtn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', () => {
      auth.signOut();
    });
  }
}

// ---------------------------
// 13) Authentication form listeners
// ---------------------------
function setupAuthFormListeners() {
  // Tab switching
  document.querySelectorAll('.auth-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.auth-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      
      const authType = tab.getAttribute('data-auth');
      document.querySelectorAll('.auth-form').forEach(form => form.classList.add('hidden'));
      document.getElementById(`${authType}Form`).classList.remove('hidden');
    });
  });

  // Login form
  document.getElementById('loginForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;

    try {
      await auth.signInWithEmailAndPassword(email, password);
    } catch (error) {
      alert(error.message);
    }
  });

  // Signup form
  document.getElementById('signupForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = document.getElementById('signupEmail').value;
    const password = document.getElementById('signupPassword').value;
    const name = document.getElementById('signupName').value;
    const username = document.getElementById('signupUsername').value.toLowerCase();

    try {
      // Validate username
      if (username.length < 3 || username.length > 20) {
        throw new Error('Username must be between 3 and 20 characters');
      }
      if (!/^[a-zA-Z0-9]+$/.test(username)) {
        throw new Error('Username can only contain letters and numbers');
      }

      // Check if username is already taken
      const usernameSnapshot = await db.collection('users')
        .where('username', '==', username)
        .limit(1)
        .get();

      if (!usernameSnapshot.empty) {
        throw new Error('This username is already taken');
      }

      const userCredential = await auth.createUserWithEmailAndPassword(email, password);
      await userCredential.user.updateProfile({
        displayName: name
      });
      
      // Create comprehensive user document in Firestore
      await db.collection('users').doc(userCredential.user.uid).set({
        displayName: name,
        username: username,
        email: email,
        phoneNumber: '',
        addresses: [],
        bio: '',
        interests: [],
        photoURL: 'default-avatar.png',
        createdAt: firebase.firestore.FieldValue.serverTimestamp(),
        updatedAt: firebase.firestore.FieldValue.serverTimestamp()
      });
    } catch (error) {
      alert(error.message);
    }
  });
}

// Helper function to render interests
function renderInterests(interests = []) {
  if (!interests || interests.length === 0) {
    return '<p class="no-interests">Add your interests</p>';
  }
  return interests.map(interest => `
    <span class="interest-tag">${interest}</span>
  `).join('');
}

// Add these new modal functions
function showEditPhoneModal(user) {
  const modalHTML = `
    <div class="modal-content">
      <h3>Edit Phone Number</h3>
      <form id="editPhoneForm">
        <div class="form-group">
          <label for="phoneNumber">Phone Number</label>
          <input type="tel" id="phoneNumber" value="${user.phoneNumber || ''}" required>
        </div>
        <div class="modal-actions">
          <button type="button" class="cancel-btn">Cancel</button>
          <button type="submit" class="save-btn">Save</button>
        </div>
      </form>
    </div>
  `;

  showModal(modalHTML, async () => {
    const phoneNumber = document.getElementById('phoneNumber').value.trim();
    try {
      await db.collection('users').doc(user.uid).update({
        phoneNumber: phoneNumber,
        updatedAt: firebase.firestore.FieldValue.serverTimestamp()
      });
      document.querySelector('.detail-value').textContent = phoneNumber;
      hideModal();
    } catch (error) {
      console.error('Error updating phone number:', error);
      alert('Failed to update phone number');
    }
  });
}

function showEditAddressScreen(user) {
  const authContainer = document.getElementById('auth-container');
  authContainer.innerHTML = `
    <div class="profile-container">
      <div class="edit-screen-header">
        <button class="back-btn">‹</button>
        <h3>Address</h3>
      </div>
      
      <form id="editAddressForm">
        <div class="edit-field-group">
          <label for="street">Street Address</label>
          <input type="text" id="street" required placeholder="123 Main St">
        </div>
        <div class="edit-field-group">
          <label for="apartment">Apartment/Suite (Optional)</label>
          <input type="text" id="apartment" placeholder="Apt 4B">
        </div>
        <div class="edit-field-group">
          <label for="city">City</label>
          <input type="text" id="city" required placeholder="City">
        </div>
        <div class="edit-field-group">
          <label for="state">State</label>
          <input type="text" id="state" required placeholder="State">
        </div>
        <div class="edit-field-group">
          <label for="zipCode">ZIP Code</label>
          <input type="text" id="zipCode" required placeholder="12345">
        </div>
        
        <button type="submit" class="save-btn">Save</button>
      </form>
    </div>
  `;

  // Pre-fill form if address exists
  if (user.addresses && Array.isArray(user.addresses) && user.addresses.length > 0) {
    const currentAddress = user.addresses[0];
    if (currentAddress) {
      document.getElementById('street').value = currentAddress.street || '';
      document.getElementById('apartment').value = currentAddress.apartment || '';
      document.getElementById('city').value = currentAddress.city || '';
      document.getElementById('state').value = currentAddress.state || '';
      document.getElementById('zipCode').value = currentAddress.zipCode || '';
    }
  }

  setupEditFormListeners('address', user);
}

function renderSavedAddresses(user) {
  if (!user.addresses || user.addresses.length === 0) {
    return '';
  }
  
  return user.addresses.map(address => `
    <div class="address-item">
      <div class="address-details">
        <div class="address-name">${address.name}</div>
        <div class="address-line">${address.street}</div>
        ${address.apartment ? `<div class="address-line">${address.apartment}</div>` : ''}
        <div class="address-line">${address.city}, ${address.state} ${address.zipCode}</div>
        <div class="address-line">United States</div>
        <div class="address-phone">${address.phone}</div>
        <div class="address-type">BILLING</div>
      </div>
      <button class="address-menu-btn">⋮</button>
    </div>
  `).join('');
}

function setupAddressListeners(user) {
  // Back button
  document.querySelector('.back-btn').addEventListener('click', () => {
    showUserProfile(user);
  });

  // Add new address
  document.querySelector('.add-address-btn').addEventListener('click', () => {
    showAddAddressForm(user);
  });

  // Address menu buttons
  document.querySelectorAll('.address-menu-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const addressItem = btn.closest('.address-item');
      const index = Array.from(addressItem.parentElement.children).indexOf(addressItem);
      showAddressOptions(user, index);
    });
  });
}

// Generic update function for simple fields
async function updateUserField(user, field, value) {
  try {
    await db.collection('users').doc(user.uid).update({
      [field]: value,
      updatedAt: firebase.firestore.FieldValue.serverTimestamp()
    });
    await showUserProfile(user); // Refresh the profile display
  } catch (error) {
    console.error(`Error updating ${field}:`, error);
    throw error;
  }
}

// ---------------------------
// 14) Address CRUD operations
// ---------------------------
async function createAddress(user, addressData) {
  try {
    const addresses = user.addresses || [];
    addresses.push(addressData);
    
    await db.collection('users').doc(user.uid).update({
      addresses: addresses,
      updatedAt: firebase.firestore.FieldValue.serverTimestamp()
    });
    
    // Update local user object
    user.addresses = addresses;
    
    // Refresh the view
    await refreshUserData(user);
    await showEditAddressScreen(user);
  } catch (error) {
    console.error('Error creating address:', error);
    throw error;
  }
}

async function updateAddress(user, index, addressData) {
  try {
    const addresses = user.addresses || [];
    addresses[index] = addressData;
    
    await db.collection('users').doc(user.uid).update({
      addresses: addresses,
      updatedAt: firebase.firestore.FieldValue.serverTimestamp()
    });
    
    user.addresses = addresses;
    await showEditAddressScreen(user);
  } catch (error) {
    console.error('Error updating address:', error);
    throw error;
  }
}

async function deleteAddress(user, index) {
  try {
    const addresses = user.addresses || [];
    addresses.splice(index, 1);
    
    await db.collection('users').doc(user.uid).update({
      addresses: addresses,
      updatedAt: firebase.firestore.FieldValue.serverTimestamp()
    });
    
    user.addresses = addresses;
    await showEditAddressScreen(user);
  } catch (error) {
    console.error('Error deleting address:', error);
    throw error;
  }
}

// Show address options menu
function showAddressOptions(user, index) {
  const addressItem = document.querySelectorAll('.address-item')[index];
  const optionsMenu = document.createElement('div');
  optionsMenu.className = 'address-options-menu';
  optionsMenu.innerHTML = `
    <button class="edit-option">Edit</button>
    <button class="delete-option">Delete</button>
  `;
  
  addressItem.appendChild(optionsMenu);
  
  // Handle option clicks
  optionsMenu.querySelector('.edit-option').addEventListener('click', () => {
    showAddAddressForm(user, index);
  });
  
  optionsMenu.querySelector('.delete-option').addEventListener('click', async () => {
    if (confirm('Are you sure you want to delete this address?')) {
      await deleteAddress(user, index);
    }
  });
  
  // Close menu when clicking outside
  document.addEventListener('click', function closeMenu(e) {
    if (!optionsMenu.contains(e.target) && !addressItem.querySelector('.address-menu-btn').contains(e.target)) {
      optionsMenu.remove();
      document.removeEventListener('click', closeMenu);
    }
  });
}

function showEditInterestsModal(user) {
  const modalHTML = `
    <div class="modal-content">
      <h3>Edit Interests</h3>
      <form id="editInterestsForm">
        <div class="form-group">
          <label for="interests">Add interests (comma-separated)</label>
          <textarea id="interests" placeholder="hiking, photography, travel...">${(user.interests || []).join(', ')}</textarea>
        </div>
        <div class="modal-actions">
          <button type="button" class="cancel-btn">Cancel</button>
          <button type="submit" class="save-btn">Save</button>
        </div>
      </form>
    </div>
  `;

  showModal(modalHTML, async () => {
    const interestsText = document.getElementById('interests').value;
    const interests = interestsText.split(',').map(i => i.trim()).filter(i => i);
    try {
      await db.collection('users').doc(user.uid).update({
        interests: interests,
        updatedAt: firebase.firestore.FieldValue.serverTimestamp()
      });
      document.querySelector('.interests-container').innerHTML = renderInterests(interests);
      hideModal();
    } catch (error) {
      console.error('Error updating interests:', error);
      alert('Failed to update interests');
    }
  });
}

// ---------------------------
// 16) Form validation
// ---------------------------
// Add these validation functions
const validators = {
  phone: {
    pattern: /^\d{10}$/,
    format: (value) => {
      const numbers = value.replace(/\D/g, '');
      if (numbers.length === 10) {
        return `(${numbers.slice(0,3)}) ${numbers.slice(3,6)}-${numbers.slice(6)}`;
      }
      return numbers;
    },
    message: 'Please enter a valid 10-digit phone number'
  },
  
  zipCode: {
    pattern: /^\d{5}(-\d{4})?$/,
    format: (value) => value.replace(/[^\d-]/g, ''),
    message: 'Enter a valid ZIP code (12345 or 12345-6789)'
  },
  
  state: {
    pattern: /^[A-Z]{2}$/,
    format: (value) => value.toUpperCase(),
    message: 'Enter a valid two-letter state code (e.g., NY)'
  }
};

// Update showEditPhoneScreen with live validation
function showEditPhoneScreen(user) {
  const authContainer = document.getElementById('auth-container');
  authContainer.innerHTML = `
    <div class="profile-container">
      <div class="edit-screen-header">
        <button class="back-btn">‹</button>
        <h3>Phone</h3>
      </div>
      
      <form id="editPhoneForm">
        <div class="edit-field-group">
          <label for="phone">Phone Number</label>
          <input type="tel" 
                 id="phone" 
                 value="${formatPhoneNumber(user.phoneNumber) || ''}"
                 placeholder="(123) 456-7890"
                 required>
          <p class="field-note">Enter your 10-digit phone number</p>
          <p class="validation-message"></p>
        </div>
        
        <button type="submit" class="save-btn">Save</button>
      </form>
    </div>
  `;

  const phoneInput = document.getElementById('phone');
  const validationMessage = document.querySelector('.validation-message');
  const saveBtn = document.querySelector('.save-btn');

  phoneInput.addEventListener('input', (e) => {
    let value = validators.phone.format(e.target.value);
    e.target.value = value;
    
    const isValid = validators.phone.pattern.test(value.replace(/\D/g, ''));
    validationMessage.textContent = isValid ? '' : validators.phone.message;
    validationMessage.className = `validation-message ${isValid ? 'valid' : 'invalid'}`;
    saveBtn.disabled = !isValid;
  });

  setupEditFormListeners('phoneNumber', user, (value) => {
    return value.replace(/\D/g, '');
  });
}

// Helper function to format phone number for display
function formatPhoneNumber(phone) {
  if (!phone) return '';
  return validators.phone.format(phone);
}

function showEditNameScreen(user) {
  const authContainer = document.getElementById('auth-container');
  authContainer.innerHTML = `
    <div class="profile-container">
      <div class="edit-screen-header">
        <button class="back-btn">‹</button>
        <h3>Name</h3>
      </div>
      
      <form id="editNameForm">
        <div class="edit-field-group">
          <label for="displayName">Name</label>
          <input type="text" id="displayName" value="${user.displayName || ''}" required>
        </div>
        
        <button type="submit" class="save-btn">Save</button>
      </form>
    </div>
  `;

  setupEditFormListeners('name', user);
}

// ---------------------------
// 15) Edit screens for each detail item
// ---------------------------
function showEditEmailScreen(user) {
  const authContainer = document.getElementById('auth-container');
  authContainer.innerHTML = `
    <div class="profile-container">
      <div class="edit-screen-header">
        <button class="back-btn">‹</button>
        <h3>Email</h3>
      </div>
      
      <form id="editEmailForm">
        <div class="edit-field-group">
          <label for="email">Email</label>
          <input type="email" id="email" value="${user.email}" disabled>
          <p class="field-note">Email cannot be changed</p>
        </div>
      </form>
    </div>
  `;

  document.querySelector('.back-btn').addEventListener('click', () => {
    showUserProfile(user);
  });
}

function showEditAddressScreen(user) {
  const authContainer = document.getElementById('auth-container');
  authContainer.innerHTML = `
    <div class="profile-container">
      <div class="edit-screen-header">
        <button class="back-btn">‹</button>
        <h3>Address</h3>
      </div>
      
      <form id="editAddressForm">
        <div class="edit-field-group">
          <label for="street">Street Address</label>
          <input type="text" id="street" required placeholder="123 Main St">
        </div>
        <div class="edit-field-group">
          <label for="apartment">Apartment/Suite (Optional)</label>
          <input type="text" id="apartment" placeholder="Apt 4B">
        </div>
        <div class="edit-field-group">
          <label for="city">City</label>
          <input type="text" id="city" required placeholder="City">
        </div>
        <div class="edit-field-group">
          <label for="state">State</label>
          <input type="text" id="state" required placeholder="State">
        </div>
        <div class="edit-field-group">
          <label for="zipCode">ZIP Code</label>
          <input type="text" id="zipCode" required placeholder="12345">
        </div>
        
        <button type="submit" class="save-btn">Save</button>
      </form>
    </div>
  `;

  // Pre-fill form if address exists
  if (user.addresses && Array.isArray(user.addresses) && user.addresses.length > 0) {
    const currentAddress = user.addresses[0];
    if (currentAddress) {
      document.getElementById('street').value = currentAddress.street || '';
      document.getElementById('apartment').value = currentAddress.apartment || '';
      document.getElementById('city').value = currentAddress.city || '';
      document.getElementById('state').value = currentAddress.state || '';
      document.getElementById('zipCode').value = currentAddress.zipCode || '';
    }
  }

  setupEditFormListeners('address', user);
}

function showEditBioScreen(user) {
  const authContainer = document.getElementById('auth-container');
  authContainer.innerHTML = `
    <div class="profile-container">
      <div class="edit-screen-header">
        <button class="back-btn">‹</button>
        <h3>Bio</h3>
      </div>
      
      <form id="editBioForm">
        <div class="edit-field-group">
          <label for="bio">Bio</label>
          <textarea id="bio" maxlength="200" placeholder="Tell us about yourself">${user.bio || ''}</textarea>
        </div>
        
        <button type="submit" class="save-btn">Save</button>
      </form>
    </div>
  `;

  setupEditFormListeners('bio', user);
}

function showEditInterestsScreen(user) {
  const authContainer = document.getElementById('auth-container');
  authContainer.innerHTML = `
    <div class="profile-container">
      <div class="edit-screen-header">
        <button class="back-btn">‹</button>
        <h3>Interests</h3>
      </div>
      
      <form id="editInterestsForm">
        <div class="edit-field-group">
          <label for="interests">Interests (comma-separated)</label>
          <textarea id="interests" placeholder="hiking, photography, travel...">${(user.interests || []).join(', ')}</textarea>
        </div>
        
        <button type="submit" class="save-btn">Save</button>
      </form>
    </div>
  `;

  setupEditFormListeners('interests', user, (value) => {
    return value.split(',').map(item => item.trim()).filter(item => item);
  });
}

// Helper function to setup form listeners
function setupEditFormListeners(type, user, transformValue = null) {
  // Back button listener
  document.querySelector('.back-btn').addEventListener('click', () => {
    showUserProfile(user);
  });

  // Form submit listener
  const form = document.querySelector('form');
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    try {
      let value;
      if (type === 'address') {
        // Handle address form data
        value = {
          addresses: [{
            street: document.getElementById('street').value.trim(),
            apartment: document.getElementById('apartment').value.trim(),
            city: document.getElementById('city').value.trim(),
            state: document.getElementById('state').value.trim(),
            zipCode: document.getElementById('zipCode').value.trim()
          }]
        };
      } else {
        // Handle other fields
        value = document.querySelector('textarea, input').value.trim();
        if (transformValue) {
          value = transformValue(value);
        }
      }

      // Update the field in Firebase
      await db.collection('users').doc(user.uid).update({
        ...(type === 'address' ? value : { [type]: value }),
        updatedAt: firebase.firestore.FieldValue.serverTimestamp()
      });

      // Show updated profile
      await showUserProfile(user);
    } catch (error) {
      console.error(`Error updating ${type}:`, error);
      alert(`Failed to update ${type}`);
    }
  });
}

// Add these update functions
async function updateBio(user) {
  const bio = document.getElementById('bio').value.trim();
  try {
    await db.collection('users').doc(user.uid).update({
      bio: bio,
      updatedAt: firebase.firestore.FieldValue.serverTimestamp()
    });
    
    // Update the user object
    user.bio = bio;
    
    // Return to profile view
    showUserProfile(user);
  } catch (error) {
    console.error('Error updating bio:', error);
    throw error;
  }
}

async function updateAspirations(user) {
  const aspirations = document.getElementById('aspirations').value.trim();
  try {
    await db.collection('users').doc(user.uid).update({
      aspirations: aspirations,
      updatedAt: firebase.firestore.FieldValue.serverTimestamp()
    });
    
    // Update the user object
    user.aspirations = aspirations;
    
    // Return to profile view
    showUserProfile(user);
  } catch (error) {
    console.error('Error updating aspirations:', error);
    throw error;
  }
}

// ---------------------------
// 17) Profile photo upload
// ---------------------------
// Add these new functions after setupProfileEventListeners
async function handleProfilePhotoUpload(user, file) {
    try {
        showLoadingState('photo');
        
        // Validate file type
        if (!file.type.startsWith('image/')) {
            throw new Error('Please upload an image file');
        }

        // Validate file size (5MB max)
        if (file.size > 5 * 1024 * 1024) {
            throw new Error('File size must be less than 5MB');
        }

        // Create storage reference with a more sanitized filename
        const storageRef = storage.ref();
        const fileExtension = file.name.split('.').pop();
        const fileName = `avatar.${fileExtension}`; // Always use 'avatar' as filename to overwrite old photo
        // Update the path to match storage rules
        const photoRef = storageRef.child(`users/${user.uid}/profile/${fileName}`);
        
        // Set proper metadata
        const metadata = {
            contentType: file.type,
            customMetadata: {
                uploadedBy: user.uid,
                uploadedAt: new Date().toISOString(),
                type: 'profile-photo'
            }
        };

        try {
            // Upload file
            const snapshot = await photoRef.put(file, metadata);
            
            // Get the download URL
            const photoURL = await snapshot.ref.getDownloadURL();
            
            // Update user profile in Firebase Auth
            await user.updateProfile({
                photoURL: photoURL
            });
            
            // Update user document in Firestore
            await db.collection('users').doc(user.uid).update({
                photoURL: photoURL,
                updatedAt: firebase.firestore.FieldValue.serverTimestamp()
            });
            
            // Update the profile photo in the UI
            const profilePhoto = document.querySelector('.profile-photo');
            if (profilePhoto) {
                profilePhoto.src = photoURL;
            }
            
            showToast('Profile photo updated successfully');
        } catch (uploadError) {
            console.error('Upload error:', uploadError);
            if (uploadError.code === 'storage/unauthorized') {
                throw new Error('You need to be logged in to upload photos');
            } else if (uploadError.code === 'storage/cors-error') {
                throw new Error('CORS error: Please try again later');
            } else {
                throw new Error('Failed to upload image: ' + uploadError.message);
            }
        }
        
        hideLoadingState('photo');
    } catch (error) {
        console.error('Error updating profile photo:', error);
        showToast(error.message || 'Failed to update profile photo', 'error');
        hideLoadingState('photo');
    }
}

// Add this helper function if not already present
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// Add this to your existing setupProfileEventListeners function
function setupPhotoUploadListener(user) {
    const photoUpload = document.getElementById('photoUpload');
    if (photoUpload) {
        photoUpload.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                // Validate file type and size
                if (!file.type.startsWith('image/')) {
                    alert('Please upload an image file');
                    return;
                }
                if (file.size > 5 * 1024 * 1024) { // 5MB limit
                    alert('Please upload an image smaller than 5MB');
                    return;
                }
                handleProfilePhotoUpload(user, file);
            }
        });
    }
}

// ---------------------------
// 18) Username editing
// ---------------------------
// Add the new showEditUsernameScreen function
async function showEditUsernameScreen(user) {
  try {
    // Get latest user data from Firestore
    const userData = await refreshUserData(user);
    if (!userData) {
      throw new Error('Could not load user data');
    }

    const authContainer = document.getElementById('auth-container');
    authContainer.innerHTML = `
      <div class="profile-container">
        <div class="edit-screen-header">
          <button class="back-btn">‹</button>
          <h3>Username</h3>
        </div>
        
        <form id="editUsernameForm">
          <div class="edit-field-group">
            <label for="username">Username</label>
            <div class="input-wrapper">
              <span class="at-symbol">@</span>
              <input type="text" 
                     id="username" 
                     value="${userData.username || userData.email.split('@')[0]}"
                     pattern="[a-zA-Z0-9]{3,20}"
                     required>
            </div>
            <p class="field-note">Username must be 3-20 characters and can only contain letters and numbers</p>
            <p class="validation-message"></p>
          </div>
          
          <button type="submit" class="save-btn">Save</button>
        </form>
      </div>
    `;

    const form = document.getElementById('editUsernameForm');
    const usernameInput = document.getElementById('username');
    const validationMessage = document.querySelector('.validation-message');
    const saveBtn = document.querySelector('.save-btn');

    // Back button listener
    document.querySelector('.back-btn').addEventListener('click', () => {
      showUserProfile(user);
    });

    // Form submission
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      
      const username = usernameInput.value.trim();
      
      // Basic validation
      if (username.length < 3 || username.length > 20) {
        validationMessage.textContent = 'Username must be between 3 and 20 characters';
        return;
      }
      
      if (!/^[a-zA-Z0-9]+$/.test(username)) {
        validationMessage.textContent = 'Username can only contain letters and numbers';
        return;
      }

      try {
        // Check username availability
        const snapshot = await db.collection('users')
          .where('username', '==', username.toLowerCase())
          .limit(1)
          .get();

        if (!snapshot.empty && snapshot.docs[0].id !== user.uid) {
          validationMessage.textContent = 'This username is already taken';
          return;
        }

        // Save the username
        await db.collection('users').doc(user.uid).update({
          username: username.toLowerCase(),
          updatedAt: firebase.firestore.FieldValue.serverTimestamp()
        });

        showToast('Username updated successfully');
        showUserProfile(user);
      } catch (error) {
        console.error('Error:', error);
        validationMessage.textContent = 'An error occurred. Please try again.';
      }
    });
  } catch (error) {
    console.error('Error loading user data:', error);
    showToast('Error loading user data. Please try again.', 'error');
  }
}

// Add these functions after showEditInterestsScreen
function showEditInterestedCategoriesScreen(user) {
  const authContainer = document.getElementById('auth-container');
  authContainer.innerHTML = `
    <div class="profile-container">
      <div class="edit-screen-header">
        <button class="back-btn">‹</button>
        <h3>Interested Categories</h3>
      </div>
      
      <div class="category-list">
        ${window.categories.map(category => `
          <div class="category-item ${user.interestedCategories?.includes(category.id) ? 'selected' : ''}" 
               data-id="${category.id}">
            <span class="category-symbol">${category.symbol}</span>
            <span class="category-name">${category.name}</span>
          </div>
        `).join('')}
      </div>
      
      <div class="form-actions">
        <button class="save-btn">Save</button>
      </div>
    </div>
  `;

  // Back button listener
  document.querySelector('.back-btn').addEventListener('click', () => {
    showUserProfile(user);
  });

  // Category selection
  document.querySelectorAll('.category-item').forEach(item => {
    item.addEventListener('click', () => {
      item.classList.toggle('selected');
    });
  });

  // Save button listener
  document.querySelector('.save-btn').addEventListener('click', async () => {
    const selectedCategories = Array.from(document.querySelectorAll('.category-item.selected'))
      .map(item => item.dataset.id);

    try {
      await db.collection('users').doc(user.uid).update({
        interestedCategories: selectedCategories,
        updatedAt: firebase.firestore.FieldValue.serverTimestamp()
      });
      showToast('Categories updated successfully');
      showUserProfile(user);
    } catch (error) {
      console.error('Error updating categories:', error);
      showToast('Failed to update categories', 'error');
    }
  });
}

function showEditInterestedTagsScreen(user) {
  const authContainer = document.getElementById('auth-container');
  authContainer.innerHTML = `
    <div class="profile-container">
      <div class="edit-screen-header">
        <button class="back-btn">‹</button>
        <h3>Interested Tags</h3>
      </div>
      
      <div class="tag-list">
        ${window.tags.map(tag => `
          <div class="tag-item ${user.interestedTags?.includes(tag.id) ? 'selected' : ''}" 
               data-id="${tag.id}">
            <span class="tag-symbol">${tag.symbol}</span>
            <span class="tag-name">${tag.name}</span>
          </div>
        `).join('')}
      </div>
      
      <div class="form-actions">
        <button class="save-btn">Save</button>
      </div>
    </div>
  `;

  // Back button listener
  document.querySelector('.back-btn').addEventListener('click', () => {
    showUserProfile(user);
  });

  // Tag selection
  document.querySelectorAll('.tag-item').forEach(item => {
    item.addEventListener('click', () => {
      item.classList.toggle('selected');
    });
  });

  // Save button listener
  document.querySelector('.save-btn').addEventListener('click', async () => {
    const selectedTags = Array.from(document.querySelectorAll('.tag-item.selected'))
      .map(item => item.dataset.id);

    try {
      await db.collection('users').doc(user.uid).update({
        interestedTags: selectedTags,
        updatedAt: firebase.firestore.FieldValue.serverTimestamp()
      });
      showToast('Tags updated successfully');
      showUserProfile(user);
    } catch (error) {
      console.error('Error updating tags:', error);
      showToast('Failed to update tags', 'error');
    }
  });
} 