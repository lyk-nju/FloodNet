/**
 * Skeleton visualization with capsule bones and sphere joints
 * Based on HumanML3D skeleton structure (22 joints)
 */

class Skeleton3D {
    constructor(scene) {
        this.scene = scene;
        this.joints = [];
        this.bones = [];
        
        // Trail settings
        this.trailPoints = [];  // Store root positions
        this.maxTrailPoints = 200;  // Maximum trail length
        this.trailLine = null;
        this.trailGeometry = null;
        this.trailMaterial = null;
        
        // HumanML3D skeleton chains (from render_skeleton.py)
        this.chains = [
            [0, 2, 5, 8, 11],      // Chain 0: spine
            [0, 1, 4, 7, 10],      // Chain 1: left leg  
            [0, 3, 6, 9, 12, 15],  // Chain 2: right leg + torso
            [9, 14, 17, 19, 21],   // Chain 3: left arm
            [9, 13, 16, 18, 20],   // Chain 4: right arm
        ];
        
        // Convert chains to bone connections
        this.boneConnections = [];
        for (const chain of this.chains) {
            for (let i = 0; i < chain.length - 1; i++) {
                this.boneConnections.push([chain[i], chain[i + 1]]);
            }
        }
        
        // Bone and joint sizes - stick figure style (thin and small)
        this.boneRadius = 0.015;  // Thin cylinder
        this.jointSize = 0.03;    // Small sphere
        
        // Colors from render_skeleton.py
        this.chainColors = [
            0xFEB21A,  // orange (chain 0 - spine)
            0x00AAFF,  // cyan (chain 1 - left leg)
            0x134686,  // aquamarine (chain 2 - right leg)
            0xFFB600,  // amber (chain 3 - left arm)
            0x00D47E,  // aquamarine (chain 4 - right arm)
        ];
        
        // Joint color: teal (0, 128, 157)
        this.jointMaterial = new THREE.MeshStandardMaterial({
            color: 0x00809D,  // teal
            metalness: 0.2,
            roughness: 0.5,
        });
        
        // Will create multiple bone materials for different chains
        this.boneMaterials = this.chainColors.map(color => 
            new THREE.MeshStandardMaterial({
                color: color,
                metalness: 0.2,
                roughness: 0.5,
            })
        );
        
        this.initSkeleton();
        this.initTrail();
    }
    
    initTrail() {
        // Create trail line with gradient opacity
        this.trailGeometry = new THREE.BufferGeometry();
        
        // Initialize with empty arrays
        const positions = new Float32Array(this.maxTrailPoints * 3);
        const colors = new Float32Array(this.maxTrailPoints * 4); // RGBA
        
        this.trailGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        this.trailGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 4));
        
        this.trailMaterial = new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 1.0,
            linewidth: 2,
        });
        
        this.trailLine = new THREE.Line(this.trailGeometry, this.trailMaterial);
        this.trailLine.frustumCulled = false;
        this.scene.add(this.trailLine);
    }
    
    initSkeleton() {
        // Create 22 joint spheres - uniform small spheres
        const jointGeometry = new THREE.SphereGeometry(this.jointSize, 16, 16);
        
        for (let i = 0; i < 22; i++) {
            const joint = new THREE.Mesh(jointGeometry, this.jointMaterial);
            joint.castShadow = true;
            joint.receiveShadow = true;
            this.joints.push(joint);
            this.scene.add(joint);
        }
        
        // Create bone cylinders - different color for each chain
        let boneIndex = 0;
        for (let chainIdx = 0; chainIdx < this.chains.length; chainIdx++) {
            const chain = this.chains[chainIdx];
            const material = this.boneMaterials[chainIdx];
            
            for (let i = 0; i < chain.length - 1; i++) {
                const bone = this.createBone(material);
                this.bones.push(bone);
                this.scene.add(bone);
                boneIndex++;
            }
        }
    }
    
    createBone(material) {
        // Create a simple thin cylinder with specified material
        const geometry = new THREE.CylinderGeometry(this.boneRadius, this.boneRadius, 1, 8);
        const bone = new THREE.Mesh(geometry, material);
        bone.castShadow = true;
        bone.receiveShadow = true;
        return bone;
    }
    
    updatePose(jointPositions) {
        /**
         * Update skeleton with new joint positions
         * jointPositions: array of shape [22, 3]
         */
        if (!jointPositions || jointPositions.length !== 22) {
            console.error('Invalid joint positions:', jointPositions);
            return;
        }
        
        // Update joint positions
        for (let i = 0; i < 22; i++) {
            const pos = jointPositions[i];
            this.joints[i].position.set(pos[0], pos[1], pos[2]);
        }
        
        // Update bone positions and orientations
        for (let i = 0; i < this.boneConnections.length; i++) {
            const [startIdx, endIdx] = this.boneConnections[i];
            const startPos = new THREE.Vector3(
                jointPositions[startIdx][0],
                jointPositions[startIdx][1],
                jointPositions[startIdx][2]
            );
            const endPos = new THREE.Vector3(
                jointPositions[endIdx][0],
                jointPositions[endIdx][1],
                jointPositions[endIdx][2]
            );
            
            this.updateBone(this.bones[i], startPos, endPos);
        }
        
        // Update trail with root position (joint 0)
        this.updateTrail(jointPositions[0]);
    }
    
    updateTrail(rootPos) {
        // Add new point to trail (project to ground y=0.01)
        const trailPoint = {
            x: rootPos[0],
            y: 0.01,  // Slightly above ground to avoid z-fighting
            z: rootPos[2]
        };
        
        // Only add if moved significantly (avoid duplicate points)
        if (this.trailPoints.length === 0) {
            this.trailPoints.push(trailPoint);
        } else {
            const lastPoint = this.trailPoints[this.trailPoints.length - 1];
            const dist = Math.sqrt(
                Math.pow(trailPoint.x - lastPoint.x, 2) + 
                Math.pow(trailPoint.z - lastPoint.z, 2)
            );
            if (dist > 0.02) {  // Minimum distance threshold
                this.trailPoints.push(trailPoint);
            }
        }
        
        // Limit trail length
        if (this.trailPoints.length > this.maxTrailPoints) {
            this.trailPoints.shift();
        }
        
        // Update geometry
        const positions = this.trailGeometry.attributes.position.array;
        const colors = this.trailGeometry.attributes.color.array;
        
        const numPoints = this.trailPoints.length;
        
        for (let i = 0; i < this.maxTrailPoints; i++) {
            if (i < numPoints) {
                const point = this.trailPoints[i];
                positions[i * 3] = point.x;
                positions[i * 3 + 1] = point.y;
                positions[i * 3 + 2] = point.z;
                
                // Gradient: older points (lower index) are more transparent
                const alpha = i / (numPoints - 1);  // 0 (oldest) to 1 (newest)
                const opacity = Math.pow(alpha, 1.5) * 0.8;  // Fade out older points
                
                // Use cyan color (matching joint color)
                colors[i * 4] = 0.0;      // R
                colors[i * 4 + 1] = 0.67; // G (cyan)
                colors[i * 4 + 2] = 0.85; // B
                colors[i * 4 + 3] = opacity; // A
            } else {
                // Hide unused vertices
                positions[i * 3] = 0;
                positions[i * 3 + 1] = 0;
                positions[i * 3 + 2] = 0;
                colors[i * 4 + 3] = 0;
            }
        }
        
        this.trailGeometry.attributes.position.needsUpdate = true;
        this.trailGeometry.attributes.color.needsUpdate = true;
        this.trailGeometry.setDrawRange(0, numPoints);
    }
    
    clearTrail() {
        this.trailPoints = [];
        this.trailGeometry.setDrawRange(0, 0);
    }
    
    updateBone(bone, startPos, endPos) {
        /**
         * Update a bone's position, rotation, and scale to connect two joints
         */
        const direction = new THREE.Vector3().subVectors(endPos, startPos);
        const length = direction.length();
        
        if (length < 0.001) return;
        
        // Position bone at midpoint
        const midpoint = new THREE.Vector3().addVectors(startPos, endPos).multiplyScalar(0.5);
        bone.position.copy(midpoint);
        
        // Scale bone to match distance
        bone.scale.y = length;
        
        // Rotate bone to point from start to end
        bone.quaternion.setFromUnitVectors(
            new THREE.Vector3(0, 1, 0),
            direction.normalize()
        );
    }
    
    setVisible(visible) {
        this.joints.forEach(joint => joint.visible = visible);
        this.bones.forEach(bone => bone.visible = visible);
        if (this.trailLine) this.trailLine.visible = visible;
    }
    
    dispose() {
        // Clean up resources
        this.joints.forEach(joint => {
            this.scene.remove(joint);
            joint.geometry.dispose();
        });
        
        this.bones.forEach(bone => {
            bone.children.forEach(child => {
                if (child.geometry) child.geometry.dispose();
            });
            this.scene.remove(bone);
        });
        
        if (this.trailLine) {
            this.scene.remove(this.trailLine);
            this.trailGeometry.dispose();
            this.trailMaterial.dispose();
        }
        
        this.jointMaterial.dispose();
        this.boneMaterials.forEach(mat => mat.dispose());
    }
}

// Export for use in main.js
window.Skeleton3D = Skeleton3D;

