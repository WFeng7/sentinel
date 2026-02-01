/**
 * Frontend Integration for Hybrid EC2-Local Camera System
 * Add this to your frontend to intelligently route camera processing.
 */

// Hybrid Camera Manager for Frontend
class HybridCameraManager {
    constructor() {
        this.apiBase = 'http://localhost:8000';
        this.cache = new Map();
        this.cacheTimeout = 60000; // 1 minute
    }

    // Get all cameras with EC2 status
    async getAllCameras() {
        try {
            const response = await fetch(`${this.apiBase}/cameras?include_ec2=true`);
            const data = await response.json();
            return data.cameras;
        } catch (error) {
            console.error('Error fetching cameras:', error);
            return [];
        }
    }

    // Get processing status for a specific camera
    async getCameraStatus(cameraId) {
        try {
            const response = await fetch(`${this.apiBase}/cameras/${cameraId}/processing-status`);
            return await response.json();
        } catch (error) {
            console.error(`Error getting status for ${cameraId}:`, error);
            return null;
        }
    }

    // Start camera processing (prefers EC2)
    async startCameraProcessing(cameraId, forceLocal = false) {
        try {
            const response = await fetch(`${this.apiBase}/cameras/${cameraId}/start?force_local=${forceLocal}`, {
                method: 'POST'
            });
            return await response.json();
        } catch (error) {
            console.error(`Error starting ${cameraId}:`, error);
            return null;
        }
    }

    // Get EC2-only cameras
    async getEC2Cameras() {
        try {
            const response = await fetch(`${this.apiBase}/ec2/cameras`);
            const data = await response.json();
            return data.ec2_only;
        } catch (error) {
            console.error('Error fetching EC2 cameras:', error);
            return [];
        }
    }

    // Get hybrid status overview
    async getHybridStatus() {
        try {
            const response = await fetch(`${this.apiBase}/cameras/hybrid-status`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching hybrid status:', error);
            return null;
        }
    }

    // Smart camera processing - checks EC2 first, falls back to local
    async ensureCameraProcessing(cameraId, options = {}) {
        const { forceLocal = false, timeout = 10000 } = options;

        // Check current status
        const status = await this.getCameraStatus(cameraId);
        if (!status) {
            throw new Error(`Could not get status for camera ${cameraId}`);
        }

        // If already running, return status
        if (status.status === 'already_running') {
            return status;
        }

        // Start processing with smart routing
        const result = await this.startCameraProcessing(cameraId, forceLocal);
        
        if (result.status === 'started') {
            console.log(`✅ Camera ${cameraId} started on ${result.location}`);
            return result;
        } else {
            throw new Error(`Failed to start camera ${cameraId}: ${result.error || 'Unknown error'}`);
        }
    }
}

export default HybridCameraManager;
