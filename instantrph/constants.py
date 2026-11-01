"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

Display constants in SI units: SLM resolution, pixel pitch and the RGB laser wavelengths.
"""

R = 800  # SLM rows
C = 1280  # SLM columns

NM = 1e-9
PITCH_M = 10.8e-6
WAVELENGTHS_M = (660 * NM, 520 * NM, 450 * NM)
