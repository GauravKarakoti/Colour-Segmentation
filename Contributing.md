# Contributing Guidelines

We welcome contributions to improve these computer vision tools! Here's how to help:

## Getting Started

1. Fork the repository
2. Clone your fork:

```bash
git clone https://github.com/yourusername/color-segmentation.git
```

3. Create a feature branch:

```bash
git checkout -b feature/your-feature
```

# Build Commands

## GUI-based visualization via X11 forwarding on Windows

### **Prerequisites**

To run OpenCV GUI applications inside Docker on Windows, install **VcXsrv**:  
🔗 **Download VcXsrv**: [VcXsrv Windows X Server](https://sourceforge.net/projects/vcxsrv/)

### **VcXsrv Setup:**

1. **Install** VcXsrv after downloading.
2. **Launch** `XLaunch` and configure:
   - Select **Multiple windows** → Click _Next_
   - Choose **Start no client** → Click _Next_
   - Enable **Disable Access Control** (important) → Click _Next_
   - Click **Finish** to start the X server

---

## **Build the Docker Image**

```sh
docker build -t segmentation-app .
```

---

## **Run the Container**

```sh
docker run --rm -it segmentation-app
```

---

## **Possible Commands**

To run a specific script

```sh
docker run --rm -it segmentation-app

docker run --rm -it segmentation-app img_segment.py

docker run --rm -it segmentation-app segment.py
```

## Development Process

- Follow PEP8 coding standards
- Keep code modular and maintainable
- Add comments for complex logic
- Update documentation when making changes

## Testing

1. Test all scripts with various inputs
2. Verify trackbar functionality
3. Check for memory leaks in video processing

## Submitting Changes

1. Commit with descriptive messages
2. Push to your feature branch
3. Open a Pull Request with:
   - Description of changes
   - Screenshots if applicable
   - Any known limitations

## Code of Conduct

- Be respectful and inclusive
- Keep discussions focused on technical topics
- No harassment or offensive language

## Reporting Issues

Include in bug reports:

1. Steps to reproduce
2. Expected vs actual behavior
3. System/environment details
4. Screenshots if applicable
