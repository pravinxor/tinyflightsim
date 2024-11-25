const Mat = {
  generateNormals: function(vertices, indices) {
    const normalSums = Array(vertices.length / 4).fill(null).map(() => [0, 0, 0]);
    const normalCounts = Array(vertices.length / 4).fill(0);

    for (let i = 0; i < indices.length; i += 3) {
      const idx1 = indices[i];
      const idx2 = indices[i + 1];
      const idx3 = indices[i + 2];

      const v1 = vertices.slice(idx1 * 4, idx1 * 4 + 3);
      const v2 = vertices.slice(idx2 * 4, idx2 * 4 + 3);
      const v3 = vertices.slice(idx3 * 4, idx3 * 4 + 3);

      const edge1 = this.sub(v2, v1);
      const edge2 = this.sub(v3, v1);

      const normal = this.normalize(this.cross(edge1, edge2));

      [idx1, idx2, idx3].forEach(idx => {
        normalSums[idx] = normalSums[idx].map((val, i) => val + normal[i]);
        normalCounts[idx]++;
      });
    }

    const normals = Array(vertices.length / 4 * 3);
    for (let i = 0; i < normalSums.length; i++) {
      const avgNormal = this.mul(normalSums[i], 1 / normalCounts[i]);

      const normalizedNormal = this.normalize(avgNormal);
      normals[i * 3 + 0] = normalizedNormal[0];
      normals[i * 3 + 1] = normalizedNormal[1];
      normals[i * 3 + 2] = normalizedNormal[2];
    }

    return normals;
  },

  generateViewMatrix: function(eye, center, up) {
    const f = this.normalize(this.sub(center, eye));
    const r = this.normalize(this.cross(f, up));
    const u = this.cross(r, f);
    return this.transpose([
      [r[0], u[0], -f[0], 0],
      [r[1], u[1], -f[1], 0],
      [r[2], u[2], -f[2], 0],
      [-this.dot(r, eye), -this.dot(u, eye), this.dot(f, eye), 1]
    ]);
  },

  generatePerspectiveMatrix: function(aspect_ratio, fov, near, far) {
    fov *= Math.PI / 180; // convert to radians
    const f = 1 / Math.tan(fov / 2);
    const rangeInv = 1 / (near - far);

    return this.transpose([
      [f / aspect_ratio, 0, 0, 0],
      [0, f, 0, 0],
      [0, 0, (near + far) * rangeInv, -1],
      [0, 0, near * far * rangeInv * 2, 0]
    ]);
  },

  transform: function(translate, scale, rotate) {
    if (translate == null) translate = [0, 0, 0];
    if (scale == null) scale = [1, 1, 1];
    if (rotate == null) rotate = [0, 0, 0];

    const toRadians = angle => angle * (Math.PI / 180);
    const [yaw, pitch, roll] = rotate.map(toRadians);

    const cosYaw = Math.cos(yaw);
    const sinYaw = Math.sin(yaw);
    const cosPitch = Math.cos(pitch);
    const sinPitch = Math.sin(pitch);
    const cosRoll = Math.cos(roll);
    const sinRoll = Math.sin(roll);

    const [tx, ty, tz] = translate;
    const [sx, sy, sz] = scale;

    return this.transpose([
      [sx * (cosYaw * cosRoll + sinYaw * sinPitch * sinRoll), sx * (sinRoll * cosPitch), sx * (-sinYaw * cosRoll + cosYaw * sinPitch * sinRoll), 0],
      [sy * (-cosYaw * sinRoll + sinYaw * sinPitch * cosRoll), sy * (cosRoll * cosPitch), sy * (sinRoll * sinYaw + cosYaw * sinPitch * cosRoll), 0],
      [sz * (sinYaw * cosPitch), sz * (-sinPitch), sz * (cosYaw * cosPitch), 0],
      [tx, ty, tz, 1]
    ]);
  },

  I: function(d) {
    return Array.from({ length: d }, (_, i) =>
      Array.from({ length: d }, (_, j) => (i === j ? 1 : 0))
    );
  },

  checkLen: function(u, v) {
    if (u.length != v.length) throw new Error(`len a: ${u.length} != len b: ${v.length}`);
  },

  add: function(u, v) {
    this.checkLen(u, v);
    return u.map((val, i) => val + v[i]);
  },

  sub: function(u, v) {
    this.checkLen(u, v);
    return u.map((val, i) => val - v[i]);
  },

  normalize: function(v) {
    const magnitude = Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
    return v.map(val => val / magnitude);
  },

  dot: function(u, v) {
    this.checkLen(u, v);
    return u.reduce((sum, val, i) => sum + val * v[i], 0);
  },

  transpose: function(m) {
    return m[0].map((_, colIndex) => m.map(row => row[colIndex]))
  },

  mul: function(a, b) {
    if (typeof b == "number") return this.mulScalar(a, b);

    const rowsA = a.length;
    const colsB = b[0].length;

    const rowIndices = Array.from({ length: rowsA }, (_, i) => i);
    const colIndices = Array.from({ length: colsB }, (_, j) => j);

    const getColumn = (matrix, j) => matrix.map(row => row[j]);

    return rowIndices.map(i =>
      colIndices.map(j =>
        this.dot(a[i], getColumn(b, j))
      )
    );
  },


  mulScalar: function(m, s) {
    return m.map(e => Array.isArray(e) ? this.mulScalar(e, s) : e * s);
  },

  cross: function(u, v) {
    if (u.length != 3 || v.length != 3) {
      throw new Error("Cross product is only defined for 3-dimensional vectors");
    }
    return [
      u[1] * v[2] - u[2] * v[1],
      u[2] * v[0] - u[0] * v[2],
      u[0] * v[1] - u[1] * v[0]
    ];
  },
};

function initShader(gl, shaderElementId, shaderType) {
  const shaderElement = document.getElementById(shaderElementId);
  if (!shaderElement) throw new Error(`Shader Element: ${shaderElement} was not found`);

  const shader = gl.createShader(shaderType);
  gl.shaderSource(shader, shaderElement.textContent.trim());
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(`Failed to compile '${shaderElement}' due to: ${gl.getShaderInfoLog(shader)}`);
  return shader;
}

function compileProgram(gl, vertexShaderId, fragmentShaderId) {
  const program = gl.createProgram();

  if (vertexShaderId != null) gl.attachShader(program, initShader(gl, vertexShaderId, gl.VERTEX_SHADER));
  if (fragmentShaderId != null) gl.attachShader(program, initShader(gl, fragmentShaderId, gl.FRAGMENT_SHADER));

  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(`Failed to compile program due to: ${gl.getProgramInfoLog(program)}`);

  return program;
}

function loadGLBuffer(gl, program, attrib, nComponents, bufferType, buffer) {
  const glBuffer = gl.createBuffer();
  gl.bindBuffer(bufferType, glBuffer);
  gl.bufferData(bufferType, buffer, gl.STATIC_DRAW);

  if (bufferType == gl.ARRAY_BUFFER) {
    const attribLocation = gl.getAttribLocation(program, attrib)
    gl.vertexAttribPointer(attribLocation, nComponents, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(attribLocation);
  }
}

function indicesConcate(arrays, bumps) {
  let cumBump = 0;
  let cum = [];

  for (let i = 0; i < arrays.length; ++i) {
    cum.push(...arrays[i].map(z => z + cumBump))
    cumBump += bumps[i];
  }
  return cum;
}

const Models = {
  MountainMesh: class {
    generateHeight(x, troughWidth, peakHeight) {
      const normalizedX = Math.abs(x); // absolute value for symmetry
      let height;

      if (normalizedX < troughWidth) {
        // in the trough area
        height = Math.random() * 0.1;
      } else {
        // outside trough
        const distanceFromTrough = normalizedX - troughWidth;
        const heightFactor = distanceFromTrough / (1.0 - troughWidth);
        height = Math.random() * peakHeight * heightFactor;
      }

      return height;
    }

    toRGB(hue, saturation, lightness) {
      const c = saturation * lightness;
      const x_temp = c * (1 - Math.abs((hue * 6) % 2 - 1));
      const m = lightness - c;

      let r, g, b;
      if (hue < 1 / 6) {
        [r, g, b] = [c, x_temp, 0];
      } else if (hue < 2 / 6) {
        [r, g, b] = [x_temp, c, 0];
      } else if (hue < 3 / 6) {
        [r, g, b] = [0, c, x_temp];
      } else if (hue < 4 / 6) {
        [r, g, b] = [0, x_temp, c];
      } else if (hue < 5 / 6) {
        [r, g, b] = [x_temp, 0, c];
      } else {
        [r, g, b] = [c, 0, x_temp];
      }
      return [r + m, g + m, b + m];
    }

    constructor(gl, program, troughWidth = 0.4, peakHeight = 0.8, nLengthPoints = 20, nWidthPoints = 20) {
      const vertices = [];
      const indices = [];
      const colors = [];

      const lengthStep = 2.0 / (nLengthPoints - 1);
      const widthStep = 2.0 / (nWidthPoints - 1);

      for (let i = 0; i < nLengthPoints; i++) {
        const y = i * lengthStep - 1;

        for (let j = 0; j < nWidthPoints; j++) {
          const x = j * widthStep - 1;
          const z = this.generateHeight(x, troughWidth, peakHeight);

          vertices.push(x, y, z, 1.0);

          const baseHue = 0.8; // purple/pink base
          const saturation = 0.6 + Math.random() * 0.2;
          const brightness = 0.5 + (z / peakHeight) * 0.5;

          const [r, g, b] = this.toRGB(baseHue, saturation, brightness);
          colors.push(r, g, b, 1.0);
        }
      }

      for (let i = 0; i < nLengthPoints - 1; i++) {
        for (let j = 0; j < nWidthPoints - 1; j++) {
          const topLeft = i * nWidthPoints + j;
          const topRight = topLeft + 1;
          const bottomLeft = (i + 1) * nWidthPoints + j;
          const bottomRight = bottomLeft + 1;

          // first triangle
          indices.push(topLeft, bottomLeft, topRight);
          // second triangle
          indices.push(topRight, bottomLeft, bottomRight);
        }
      }

      const normals = Mat.generateNormals(vertices, indices);

      this.indicesLength = indices.length;
      this.vao = gl.createVertexArray();
      gl.bindVertexArray(this.vao);

      loadGLBuffer(gl, program, "aPosition", 4, gl.ARRAY_BUFFER, Float32Array.from(vertices));
      loadGLBuffer(gl, program, "aColor", 4, gl.ARRAY_BUFFER, Float32Array.from(colors));
      loadGLBuffer(gl, program, "aNormal", 3, gl.ARRAY_BUFFER, Float32Array.from(normals));
      loadGLBuffer(gl, program, "", null, gl.ELEMENT_ARRAY_BUFFER, Uint16Array.from(indices));

      gl.bindVertexArray(null);
    }

    draw(gl) {
      gl.bindVertexArray(this.vao);
      gl.drawElements(gl.TRIANGLES, this.indicesLength, gl.UNSIGNED_SHORT, 0);
    }

    drawWireframe(gl) {
      gl.bindVertexArray(this.vao);
      gl.drawElements(gl.LINES, this.indicesLength, gl.UNSIGNED_SHORT, 0);
    }
  },

  Circle: class {
    constructor(gl, program, radius = 1.0, segments = 32, startAngle = 0, endAngle = 2 * Math.PI) {
      const vertices = [];
      const colors = [];
      const indices = [];

      const baseColor = [1.0, 1.0, 1.0, 1.0];

      const interpolateColor = (y) => {
        const t = (y + radius) / (2 * radius);

        const gradLevel = 10;
        return Mat.add(baseColor, [1.0, (t + gradLevel) / gradLevel, (t + gradLevel) / gradLevel, 1.0])
      };

      vertices.push(0, 0, 0, 1.0);
      colors.push(...interpolateColor(0));

      for (let i = 0; i <= segments; i++) {
        const angleRange = endAngle - startAngle;
        const angleStep = angleRange / segments;

        const angle = startAngle + (i * angleStep);

        const x = radius * Math.cos(angle);
        const z = radius * Math.sin(angle);

        vertices.push(x, 0, z, 1.0);
        colors.push(...interpolateColor(z));

        if (i > 0) {
          indices.push(0, i, i + 1);
        }
      }

      this.indicesLength = indices.length;
      this.vao = gl.createVertexArray();
      gl.bindVertexArray(this.vao);

      loadGLBuffer(gl, program, "aColor", 4, gl.ARRAY_BUFFER, Float32Array.from(colors));
      loadGLBuffer(gl, program, "aPosition", 4, gl.ARRAY_BUFFER, Float32Array.from(vertices));
      loadGLBuffer(gl, program, "", null, gl.ELEMENT_ARRAY_BUFFER, Uint16Array.from(indices));

      gl.bindVertexArray(null);
    }

    draw(gl) {
      gl.bindVertexArray(this.vao);
      gl.drawElements(gl.TRIANGLES, this.indicesLength, gl.UNSIGNED_SHORT, 0);
    }
  },

  JetFlap: class {
    constructor(gl, program) {
      const height = 0.08;
      const topLength = 0.03;
      const bottomLength = 0.05;

      const vertices = [
        0.0, 0.0, 0.0, 1.0,
        0.0, bottomLength, 0.0, 1.0,
        0.0, 0.0, height, 1.0,
        0.0, topLength, height, 1.0,
        0.0, topLength, 0.0, 1.0,
      ];

      const indices = [
        0, 1, 3,
        0, 2, 3,
        0, 4, 3,
      ];

      const colors = [
        0.45, 0.45, 0.55, 1.0,
        0.4, 0.4, 0.5, 1.0,
        0.5, 0.5, 0.6, 1.0,
        0.48, 0.48, 0.58, 1.0,
        0.45, 0.45, 0.55, 1.0
      ];

      const normals = Mat.generateNormals(vertices, indices);

      this.indicesLength = indices.length;
      this.vao = gl.createVertexArray();
      gl.bindVertexArray(this.vao);

      loadGLBuffer(gl, program, "aColor", 4, gl.ARRAY_BUFFER, Float32Array.from(colors))
      loadGLBuffer(gl, program, "aPosition", 4, gl.ARRAY_BUFFER, Float32Array.from(vertices));
      loadGLBuffer(gl, program, "aNormal", 3, gl.ARRAY_BUFFER, Float32Array.from(normals));
      loadGLBuffer(gl, program, "", null, gl.ELEMENT_ARRAY_BUFFER, Uint8Array.from(indices));

      gl.bindVertexArray(null);
    }

    draw(gl) {
      gl.bindVertexArray(this.vao);
      gl.drawElements(gl.TRIANGLES, this.indicesLength, gl.UNSIGNED_BYTE, 0);
    }
  },

  JetBody: class {
    constructor(gl, program) {
      const wingCenterStart = 0.05
      const wingStart = 0.5;
      const wingEnd = -0.1
      const wingSpan = 0.3;

      const centerStart = 0.75;
      const centerEnd = -0.25;
      const centerTopSpan = 0.025;
      const centerMiddleSpan = 0.075;
      const centerBottomSpan = 0.05
      const centerTopHeight = 0.015;
      const centerBottomHeight = 0.03;

      const vLeftWing = [
        -wingCenterStart, wingStart, -0.001, 1.0,
        -wingSpan, 0.0, 0.0, 1.0,
        -wingCenterStart, 0.0, 0.0, 1.0,
        -wingCenterStart, wingEnd, -0.001, 1.0
      ];
      const iLeftWing = [
        0, 1, 2,
        3, 1, 2
      ];
      const cLeftWing = [
        0.6, 0.6, 0.6, 1.0,
        0.6, 0.6, 0.65, 1.0,
        0.6, 0.6, 0.70, 1.0,
        0.6, 0.6, 0.68, 1.0
      ];

      const vRightWing = [
        wingCenterStart, wingStart, -0.001, 1.0,
        wingSpan, 0.0, 0.0, 1.0,
        wingCenterStart, 0.0, 0.0, 1.0,
        wingCenterStart, wingEnd, -0.001, 1.0,
      ];
      const iRightWing = iLeftWing;
      const cRightWing = [
        0.6, 0.6, 0.6, 1.0,
        0.6, 0.6, 0.65, 1.0,
        0.6, 0.6, 0.70, 1.0,
        0.6, 0.6, 0.68, 1.0
      ];

      const vCenter = [
        -centerBottomSpan, centerStart, -centerBottomHeight, 1.0,
        -centerBottomSpan, centerEnd, -centerBottomHeight, 1.0,
        centerBottomSpan, centerStart, -centerBottomHeight, 1.0,
        centerBottomSpan, centerEnd, -centerBottomHeight, 1.0,

        -centerMiddleSpan, centerStart, 0.0, 1.0,
        -centerMiddleSpan, centerEnd, 0.0, 1.0,
        centerMiddleSpan, centerStart, 0.0, 1.0,
        centerMiddleSpan, centerEnd, 0.0, 1.0,

        -centerTopSpan, centerStart, centerTopHeight, 1.0,
        -centerTopSpan, centerEnd, centerTopHeight, 1.0,
        centerTopSpan, centerStart, centerTopHeight, 1.0,
        centerTopSpan, centerEnd, centerTopHeight, 1.0,
      ];
      const iCenter = [
        0, 1, 3,
        0, 2, 3,

        4, 5, 7,
        4, 6, 7,

        0, 1, 5,
        0, 4, 5,

        2, 3, 7,
        2, 6, 7,

        8, 9, 11,
        8, 10, 11,

        10, 11, 7,
        10, 6, 7,

        8, 9, 5,
        8, 4, 5,
      ];
      const cCenter = [
        0.3, 0.3, 0.4, 1.0,
        0.35, 0.35, 0.45, 1.0,
        0.3, 0.3, 0.4, 1.0,
        0.35, 0.35, 0.45, 1.0,

        0.4, 0.4, 0.5, 1.0,
        0.45, 0.45, 0.55, 1.0,
        0.4, 0.4, 0.5, 1.0,
        0.45, 0.45, 0.55, 1.0,

        0.5, 0.5, 0.6, 1.0,
        0.55, 0.55, 0.65, 1.0,
        0.5, 0.5, 0.6, 1.0,
        0.55, 0.55, 0.65, 1.0
      ];

      const colors = [...cLeftWing, ...cRightWing, ...cCenter];
      const vertices = [...vLeftWing, ...vRightWing, ...vCenter];
      const indices = indicesConcate([iLeftWing, iRightWing, iCenter], [vLeftWing.length, vRightWing.length, vCenter.length].map(l => l / 4));
      const normals = Mat.generateNormals(vertices, indices);

      this.indicesLength = indices.length;
      this.vao = gl.createVertexArray();
      gl.bindVertexArray(this.vao);

      loadGLBuffer(gl, program, "aColor", 4, gl.ARRAY_BUFFER, Float32Array.from(colors))
      loadGLBuffer(gl, program, "aPosition", 4, gl.ARRAY_BUFFER, Float32Array.from(vertices));
      loadGLBuffer(gl, program, "aNormal", 3, gl.ARRAY_BUFFER, Float32Array.from(normals));
      loadGLBuffer(gl, program, "", null, gl.ELEMENT_ARRAY_BUFFER, Uint8Array.from(indices));

      gl.bindVertexArray(null);
    }

    draw(gl) {
      gl.bindVertexArray(this.vao);
      gl.drawElements(gl.TRIANGLES, this.indicesLength, gl.UNSIGNED_BYTE, 0);
    }
  }
}

const MountainControls = {
  scrollSpeed: 1.5,

  setupListeners: function() {
    const speedSlider = document.getElementById('mountain-speed');
    const speedValue = document.getElementById('mountain-speed-value');

    speedSlider.addEventListener('input', (e) => {
      const value = parseFloat(e.target.value);
      MountainControls.scrollSpeed = value;
      speedValue.textContent = `${value.toFixed(1)}x`;
    });
  }
};


class MountainTerrain {
  constructor(gl, program) {
    this.program = program != null ? program : compileProgram(gl, "vertex-shader", "fragment-shader");
    this.mountains = [];
    this.numSections = 4;
    this.sectionLength = 2;

    this.modelViewLoc = gl.getUniformLocation(this.program, "modelView");
    this.projectionLoc = gl.getUniformLocation(this.program, "projection");
    this.colorShiftLoc = gl.getUniformLocation(this.program, "colorShift");
    this.sunColorLoc = gl.getUniformLocation(this.program, "sunColor");

    this.cameraPosLoc = gl.getUniformLocation(this.program, "cameraPos");
    this.lightPosLoc = gl.getUniformLocation(this.program, "lightPos");

    for (let i = 0; i < this.numSections; i++) {
      const troughWidth = 0.1;
      const peakHeight = 2;
      const lengthPoints = 25;
      const widthPoints = 25;

      this.mountains.push(new Models.MountainMesh(
        gl,
        this.program,
        troughWidth,
        peakHeight,
        lengthPoints,
        widthPoints
      ));
    }
  }

  draw(gl, modelView, projection, scrollOffset, lightPos, cameraPos) {
    gl.useProgram(this.program);
    const baseOffset = scrollOffset % this.sectionLength;

    gl.uniformMatrix4fv(this.projectionLoc, true, projection.flat());

    gl.uniform3fv(this.lightPosLoc, Mat.normalize(lightPos));
    gl.uniform3fv(this.cameraPosLoc, Mat.normalize(cameraPos));
    gl.uniform3fv(this.sunColorLoc, SunControls.color);

    for (let i = 0; i < this.numSections; i++) {
      const sectionOffset = i * this.sectionLength - baseOffset;
      const secIndex = Math.floor((i + Math.floor(scrollOffset / this.sectionLength)) % this.numSections);

      const sectionModelView = Mat.mul(modelView, Mat.transform([0, sectionOffset + 0.8, -0.5], null, null));
      gl.uniformMatrix4fv(this.modelViewLoc, true, sectionModelView.flat());

      // draw with color shift effect
      gl.uniform3f(this.colorShiftLoc, 0.2, 0.2, 0.2);
      this.mountains[secIndex].draw(gl);
      gl.uniform3f(this.colorShiftLoc, 1, 1, 1);
      this.mountains[secIndex].drawWireframe(gl);
    }
  }
}

function hexToRGB(hex) {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  return [r, g, b];
}

const SunControls = {
  color: hexToRGB('#ffffff'),

  setupListeners: function() {
    const colorPicker = document.getElementById('sun-color');
    const colorPreview = document.getElementById('sun-color-preview');

    colorPreview.style.backgroundColor = colorPicker.value;

    colorPicker.addEventListener('input', (e) => {
      const hexColor = e.target.value;
      SunControls.color = hexToRGB(hexColor);
      colorPreview.style.backgroundColor = hexColor;
    });
  }
};

class Sun {
  constructor(gl) {
    this.program = compileProgram(gl, "sun-vertex-shader", "sun-fragment-shader");
    this.model = new Models.Circle(gl, this.program, 1, 64, 0, Math.PI);
    this.modelViewLoc = gl.getUniformLocation(this.program, "modelView");
    this.projectionLoc = gl.getUniformLocation(this.program, "projection");
    this.sunColorLoc = gl.getUniformLocation(this.program, "color");

  }

  draw(gl, modelView, projection, sunPos) {
    gl.useProgram(this.program);
    const sunModelView = Mat.mul(modelView, Mat.transform(sunPos, Mat.mul([1, 1, 1], 2), null));

    gl.uniformMatrix4fv(this.modelViewLoc, true, sunModelView.flat());
    gl.uniformMatrix4fv(this.projectionLoc, true, projection.flat());
    gl.uniform3fv(this.sunColorLoc, SunControls.color);

    this.model.draw(gl);
  }
}

const PlaneControls = {
  bodyYaw: 0,
  bodyPitch: 0,
  bodyRoll: 0,
  wingFlaps: 0,
  verticalStabilizers: 0,
  horizontalStabilizers: 0,

  setupListeners: function() {
    const controls = [
      { id: 'body-yaw', prop: 'bodyYaw' },
      { id: 'body-pitch', prop: 'bodyPitch' },
      { id: 'body-roll', prop: 'bodyRoll' },
      { id: 'wing-flaps', prop: 'wingFlaps' },
      { id: 'vertical-stabilizers', prop: 'verticalStabilizers' },
      { id: 'horizontal-stabilizers', prop: 'horizontalStabilizers' }
    ];

    controls.forEach(control => {
      const slider = document.getElementById(control.id);
      const valueDisplay = document.getElementById(`${control.id}-value`);

      slider.addEventListener("input", (e) => {
        const value = parseInt(e.target.value);
        PlaneControls[control.prop] = value;
        valueDisplay.textContent = `${value}°`;
      });
    });
  }
};

class FighterJet {
  constructor(gl, program) {
    this.program = program != null ? program : compileProgram(gl, "vertex-shader", "fragment-shader");
    this.modelViewLoc = gl.getUniformLocation(this.program, "modelView");
    this.projectionLoc = gl.getUniformLocation(this.program, "projection");

    this.cameraPosLoc = gl.getUniformLocation(this.program, "cameraPos");
    this.lightPosLoc = gl.getUniformLocation(this.program, "lightPos");
    this.colorShiftLoc = gl.getUniformLocation(this.program, "colorShift");
    this.sunColorLoc = gl.getUniformLocation(this.program, "sunColor");



    this.body = new Models.JetBody(gl, this.program);
    this.stabilizer = new Models.JetFlap(gl, this.program);
  }

  draw(gl, modelView, projection, yaw, pitch, roll, lightPos, cameraPos) {
    gl.useProgram(this.program);

    gl.uniform3fv(this.sunColorLoc, SunControls.color);
    gl.uniform3fv(this.lightPosLoc, Mat.normalize(lightPos));
    gl.uniform3fv(this.cameraPosLoc, Mat.normalize(cameraPos));
    gl.uniform3f(this.colorShiftLoc, 1, 1, 1);

    // apply body rotations from the sliders
    const totalYaw = yaw + PlaneControls.bodyYaw;
    const totalPitch = pitch + PlaneControls.bodyPitch;
    const totalRoll = roll + PlaneControls.bodyRoll;

    const rotatedModelView = Mat.mul(modelView, Mat.transform(null, null, [totalYaw, totalPitch, totalRoll]));

    gl.uniformMatrix4fv(this.projectionLoc, true, projection.flat());
    gl.uniformMatrix4fv(this.modelViewLoc, true, rotatedModelView.flat());

    this.body.draw(gl);

    const wingFlapModelViews = [
      // left wing flap
      Mat.mul(rotatedModelView, Mat.transform(
        [-0.085, -0.0025, 0],
        [1, -2, 2],
        [-90, 0, -PlaneControls.wingFlaps]
      )),
      // right wing flap
      Mat.mul(rotatedModelView, Mat.transform(
        [0.085, -0.0025, 0],
        [1, -2, 2],
        [90, 0, PlaneControls.wingFlaps]
      )),
      // left vertical stabilizer
      Mat.mul(rotatedModelView, Mat.transform(
        [-0.05, -0.2, 0],
        [1, 1.5, 1.75],
        [-15 + PlaneControls.verticalStabilizers, 0, 0]
      )),
      // right vertical stabilizer
      Mat.mul(rotatedModelView, Mat.transform(
        [0.05, -0.2, 0],
        [1, 1.5, 1.75],
        [15 + PlaneControls.verticalStabilizers, 0, 0]
      )),
      // left horizontal stabilizer
      Mat.mul(rotatedModelView, Mat.transform(
        [-0.07, -0.18, 0],
        [1, -1.5, 1.5],
        [-90, 0, -PlaneControls.horizontalStabilizers]
      )),
      // right horizontal stabilizer
      Mat.mul(rotatedModelView, Mat.transform(
        [0.07, -0.18, 0],
        [1, -1.5, -1.5],
        [-90, 0, PlaneControls.horizontalStabilizers]
      ))
    ];

    wingFlapModelViews.forEach(flapModelView => {
      gl.uniformMatrix4fv(this.modelViewLoc, true, flapModelView.flat());
      this.stabilizer.draw(gl);
    });
  }
}

const IPDControls = {
  value: 0,

  setupListeners: function() {
    const slider = document.getElementById('ipd-control');
    const valueDisplay = document.getElementById('ipd-control-value');

    slider.addEventListener('input', (e) => {
      IPDControls.value = parseFloat(e.target.value);
      valueDisplay.textContent = IPDControls.value.toFixed(3);
    });
  }
};

function renderLoop(canvas, gl) {
  let [yaw, pitch, roll] = [0, 0, 0];
  const KeyState = {
    w: false,
    s: false,
    a: false,
    d: false,

    update: function(dt) {
      const rotationSpeed = 115;
      const rt = rotationSpeed * dt;

      if (KeyState.a) yaw -= rt;
      if (KeyState.d) yaw += rt;
      if (KeyState.s) pitch += rt;
      if (KeyState.w) pitch -= rt;
    }
  };

  function setupListeners() {
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const zoomSpeed = 0.01;
      cameraPos[2] += event.deltaY * zoomSpeed;
      cameraPos[2] = Math.max(0, cameraPos[2]); // clamp camera position to prevent flipping
    });

    let mouseIsDown = false;
    canvas.addEventListener("mousedown", () => { mouseIsDown = true });
    canvas.addEventListener("mouseup", () => { mouseIsDown = false });
    canvas.onmousemove = (event) => {
      if (!mouseIsDown) return;
      const moveSpeed = 0.1;
      cameraPos[0] -= event.movementX * moveSpeed;
      cameraPos[2] += event.movementY * moveSpeed;
    };

    document.addEventListener("keydown", (event) => {
      const key = event.key.toLowerCase();
      if (KeyState.hasOwnProperty(key)) KeyState[key] = true;
    });
    document.addEventListener("keyup", (event) => {
      const key = event.key.toLowerCase();
      if (KeyState.hasOwnProperty(key)) KeyState[key] = false;
    });
  }

  function calculatePlaneDirection(yaw, pitch) {
    const yawRad = yaw * (Math.PI / 180);
    const pitchRad = pitch * (Math.PI / 180);

    return [
      Math.sin(yawRad),
      Math.cos(yawRad),
      Math.sin(pitchRad)
    ];
  }

  function clampPosition(position) {
    const [x, y, z] = [
      [-2, 2],
      [-5, 2],
      [0.2, 2.5]
    ];

    return [
      Math.max(x[0], Math.min(x[1], position[0])),
      Math.max(y[0], Math.min(y[1], position[1])),
      Math.max(z[0], Math.min(z[1], position[2]))
    ];
  }

  const defaultProgram = compileProgram(gl, "vertex-shader", "fragment-shader");
  const isLeftEyeLoc = gl.getUniformLocation(defaultProgram, "isLeftEye");

  let cameraPos = [0, -5, 1.5];

  let model = Mat.mul(Mat.I(4), 5);
  model[3] = [0, 0, 0, 1];
  model = Mat.transpose(model);

  let planePos = [0, 0, 0];
  const fj = new FighterJet(gl, defaultProgram);
  const planeSpeed = 5;

  const mountains = new MountainTerrain(gl, defaultProgram);

  let sunPos = [0, 5, -0.5];
  let lightOffset = [0, 20, -10];
  const sun = new Sun(gl, 10);

  const damping = 0.95;

  let start = 0;
  let scrollOffset = 0;

  function renderEyeView(eyePosition, isLeftEye, projection) {
    const view = Mat.generateViewMatrix(eyePosition, planePos, [0, 0, 1]);
    const modelView = Mat.mul(view, model);

    let planeModelView = Mat.transpose(model);
    planeModelView[3] = [...planePos, 1];
    planeModelView = Mat.transpose(planeModelView);
    planeModelView = Mat.mul(view, planeModelView);

    gl.uniform1i(isLeftEyeLoc, isLeftEye ? 1 : 0);

    fj.draw(gl, planeModelView, projection, yaw, pitch, roll, Mat.add(sunPos, lightOffset), eyePosition);
    mountains.draw(gl, modelView, projection, scrollOffset, Mat.add(sunPos, lightOffset), eyePosition);
    sun.draw(gl, modelView, projection, sunPos);
  }

  function drawAnaglyphScene(leftEyePos, rightEyePos) {
    const projection = Mat.generatePerspectiveMatrix(
      canvas.width / canvas.height,
      45,
      0.01,
      33
    );

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // draw from left eye (red)
    gl.colorMask(true, false, false, true);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    renderEyeView(leftEyePos, true, projection);

    // draw from right eye (cyan)
    gl.colorMask(false, true, true, true);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    renderEyeView(rightEyePos, false, projection);

    // reset color mask to avoid problems elsewheere
    gl.colorMask(true, true, true, true);
  }

  function render(timestamp) {
    const dt = (timestamp - start) / 1000;
    start = timestamp;

    KeyState.update(dt);
    [yaw, pitch, roll] = Mat.mulScalar([yaw, pitch, roll], damping);

    scrollOffset += MountainControls.scrollSpeed * dt;

    const direction = calculatePlaneDirection(yaw, pitch);
    const movement = Mat.mulScalar(direction, planeSpeed * dt);

    planePos = clampPosition(Mat.add(planePos, movement));

    const leftEyePos = Mat.add(cameraPos, [-IPDControls.value / 2, 0, 0]);
    const rightEyePos = Mat.add(cameraPos, [IPDControls.value / 2, 0, 0]);

    drawAnaglyphScene(leftEyePos, rightEyePos);
    requestAnimationFrame(render);
  }

  setupListeners();
  PlaneControls.setupListeners();
  SunControls.setupListeners();
  MountainControls.setupListeners();
  IPDControls.setupListeners();

  render(0);
}

function glWindowRefresh(gl, canvas) {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  gl.viewport(0, 0, canvas.width, canvas.height);
}

window.onload = function init() {
  const canvas = document.getElementById("gl-canvas");

  const gl = canvas.getContext("webgl2");
  if (!gl) throw new Error("WebGL 2.0 isn't available");

  window.addEventListener("resize", () => glWindowRefresh(gl, canvas));

  gl.clearColor(...Mat.mulScalar([0.2, 0.05, 0.22], 0.6), 1.0);
  glWindowRefresh(gl, canvas);

  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);

  renderLoop(canvas, gl);
}
