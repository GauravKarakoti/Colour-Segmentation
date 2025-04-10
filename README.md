<a id="top"></a>
# Color Segmentation Tools


[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/Apertre-2-0)


<table align="center">
    <thead align="center">
        <tr border: 1px;>
            <td><b><img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/tarikul-islam-anik/main/assets/images/Star.png" width="20" height="20"> Stars</b></td>
            <td><b>🍴 Forks</b></td>
            <td><b><img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/tarikul-islam-anik/main/assets/images/Lady%20Beetle.png" width="20" height="20"> Issues</b></td>
            <td><b><img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/tarikul-islam-anik/main/assets/images/Check%20Mark%20Button.png" width="20" height="20"> Open PRs</b></td>
            <td><b><img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/tarikul-islam-anik/main/assets/images/Cross%20Mark.png" width="20" height="20"> Closed PRs</b></td>
        </tr>
     </thead>
    <tbody>
         <tr>
            <td><img alt="Stars" src="https://img.shields.io/github/stars/GauravKarakoti/Colour-Segmentation?style=flat&logo=github"/></td>
             <td><img alt="Forks" src="https://img.shields.io/github/forks/GauravKarakoti/Colour-Segmentation?style=flat&logo=github"/></td>
            <td><img alt="Issues" src="https://img.shields.io/github/issues/GauravKarakoti/Colour-Segmentation?style=flat&logo=github"/></td>
            <td><img alt="Open Pull Requests" src="https://img.shields.io/github/issues-pr/GauravKarakoti/Colour-Segmentation?style=flat&logo=github"/></td>
           <td><img alt="Closed Pull Requests" src="https://img.shields.io/github/issues-pr-closed/GauravKarakoti/Colour-Segmentation?style=flat&color=critical&logo=github"/></td>
        </tr>
    </tbody>
</table>
</div>

---

<h3> <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/tarikul-islam-anik/main/assets/images/Man%20Technologist%20Light%20Skin%20Tone.png" width="50px"> Featured In</h3>
<tr>
<td align="center">
<a href="https://s2apertre.resourcio.in"><img src="https://s2apertre.resourcio.in/Logo_primary.svg" height="140px" width="180px" alt="Apertre 2025"></a>
</td>
</tr>

---

<h3><img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/tarikul-islam-anik/main/assets/images/Writing%20Hand%20Light%20Skin%20Tone.png" alt="Rocket" width="40" height="40" />Project Overview</h3>
<p style="font-family:var(--ff-philosopher);">
A suite of OpenCV-based tools for color-driven segmentation in images and videos</p>
<a href="https://colour-segmentation.streamlit.app/">Deployed Here</a>

---

## <p style="font-family:var(--ff-philosopher);font-size:3rem;text-align:center;"><img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/High%20Voltage.png" alt="High Voltage" width="40" height="40" />Tech Stack</p>
<center>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python">
  </a>
  <a href="https://opencv.org/">
    <img src="https://img.shields.io/badge/CV-Computer%20Vision-orange?style=for-the-badge&logo=opencv&logoColor=white" alt="Computer Vision">
  </a>
</center>


<br><be>

---

## Prerequisites
<p style="font-family:var(--ff-philosopher);">- Python 3.x</p>
<p>- OpenCV (`pip install opencv-python`)</p>
<p>- NumPy (`pip install numpy`)</p>

<br><be>

---

## <p style="font-size:3rem;"><img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/tarikul-islam-anik/main/assets/images/Man%20Technologist%20Light%20Skin%20Tone.png" width="50px"> Get Started</p>

### Installation

<p style="font-family:var(--ff-philosopher);">To contribute to theColour-Segmentation repository, follow these steps:</p>

1. **Fork the Repository:**
   Click on the "Fork" button on the repository's GitHub page to create a copy of the repository in your GitHub account.

2. **Clone the repository:**
   Clone the forked repository to your local machine using the following command in your terminal.
   ```bash
    git clone https://github.com/GauravKarakoti/color-segmentation.git
    cd color-segmentation
   ```
3. **Install dependencies:**
    ```bash
        pip install -r requirements.txt
    ``` 
3. **Add a remote upstream:**
   ```bash
   git remote add upstream https://github.com/GauravKarakoti/Colour-Segmentation
   ```
4. **Create a new branch:**
   Create a new branch for your changes. Run the following command in your terminal.
   ```bash
   git checkout -b <your-branch-name>
   ```
5. **Make the desired changes:**
   Make the desired changes to the source code.

6. **Add your changes:**
   Add your changes to the staging area. Run the following command in your terminal.
   ```bash
   git add <File1 changed> <File2 changed> ...
   ```
7. **Commit your changes:**
   Commit your changes with a meaningful commit message. Run the following command in your terminal.
   ```bash
   git commit -m "<your-commit-message>"
   ```
8. **Push your changes:**
   Push your changes to your forked repository. Run the following command in your terminal
   ```bash
   git push origin <your-branch-name>
   ```
9. **Create a Pull Request:**
   Go to the GitHub page of your forked repository. You should see a prompt to create a pull request (PR). Click on it, compare the changes, and create the PR.
<br><be>

---
### Running Tests Locally
Before submitting a PR, please run the following checks locally:

```bash
# Install development dependencies
pip install -r requirements.txt
pip install flake8 black pytest

# Run linting
flake8 .

# Run formatting check
black --check .

# Run tests
pytest
```


### Release Process
We follow [Semantic Versioning](https://semver.org/) for releases:
- **Major version**: Incompatible API changes
- **Minor version**: New functionality in a backward-compatible manner
- **Patch version**: Backward-compatible bug fixes

Release tags are created from the `main` branch after a successful merge from `develop`.

### Continuous Integration
Our CI pipeline automatically runs on every push and PR, checking:
- Code formatting with `black`
- Linting with `flake8`
- Unit tests with `pytest`
- Script verification to ensure all scripts run without errors


## Usage
### Image Segmentation
```bash
python img_segment.py
```
- Adjust trackbars to set HSV thresholds
- Press ESC to exit
- Ensure `image1.webp` exists in project directory or modify filename

### Color Palette
```bash
python palette.py
```
- Use RGB trackbars to mix colors
- Displays resulting color in window

### Video Segmentation
```bash
python segment.py
```
- Adjust trackbars for real-time video processing
- Ensure `video1.mp4` exists or modify video path
- Press ESC to exit

---

<h2>Project Admin:</h2>

<table>
<tr>
<td align="center">
<a href="https://github.com/GauravKarakoti"><img src="https://avatars.githubusercontent.com/u/180496085?v=4" height="140px" width="140px" alt="Gaurav Karakoti "></a><br><sub><b>Gaurav Karakoti </b><br><a href="https://www.linkedin.com/in/gaurav-karakoti-248960302/"><img src="https://github-production-user-asset-6210df.s3.amazonaws.com/73993775/278833250-adb040ea-e3ef-446e-bcd4-3e8d7d4c0176.png" width="45px" height="45px"></a></sub>
</td>
</tr>
</table>

---

<div align="center">
  <h2 style="font-size:3rem;">Our Contributors <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Red%20Heart.png" alt="Red Heart" width="40" height="40" /></h2>
  <h3>Thank you for contributing to our repository</h3>

<a href="https://github.com/GauravKarakoti/Colour-Segmentation/graphs/contributors">
<img src="https://contributors-img.web.app/image?repo=GauravKarakoti/Colour-Segmentation"/>

  </a>

<p style="font-family:var(--ff-philosopher);font-size:3rem;"><b> Show some <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Red%20Heart.png" alt="Red Heart" width="40" height="40" /> by starring this awesome repository!

</div>
<center>
<h3 style="font-size:2rem;">
If you find this project helpful, please consider giving it a star! <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/tarikul-islam-anik/main/assets/images/Star.png" width="30" height="30"></p>
</center>

[🔝 Back to Top](#top)
