import numpy as np
from scipy.integrate import simpson

class MotorRefletor:
    def __init__(self, D, f_over_d, freq_ghz):
        self.D = D
        self.f = f_over_d * D
        self.freq = freq_ghz * 1e9
        self.c = 3e8
        self.wavelength = self.c / self.freq
        self.k = 2 * np.pi / self.wavelength

    def calcular_sistema(self, delta_x=0.0, q=2.5):
        """ [Mantém a função da Aba 1 idêntica] """
        rho = np.linspace(0, self.D / 2, 80)
        phi = np.linspace(0, 2 * np.pi, 80)
        Rho, Phi = np.meshgrid(rho, phi)
        X = Rho * np.cos(Phi)
        Y = Rho * np.sin(Phi)
        Z = (X**2 + Y**2) / (4 * self.f)

        R_f = np.sqrt((X - delta_x)**2 + Y**2 + (self.f - Z)**2)
        theta_f = np.arccos((self.f - Z) / R_f)
        E_inc = (np.cos(theta_f) ** q) / R_f * np.exp(-1j * self.k * R_f)
        E_aperture = E_inc

        theta_deg = np.linspace(-20, 20, 250)
        theta_rad = np.deg2rad(theta_deg)
        E_far = np.zeros_like(theta_rad, dtype=complex)

        for i, t in enumerate(theta_rad):
            phase_factor = np.exp(1j * self.k * X * np.sin(t))
            integrand = E_aperture * phase_factor * Rho
            int_phi = simpson(integrand, x=np.linspace(0, 2*np.pi, integrand.shape[0]), axis=0)
            E_far[i] = simpson(int_phi, x=np.linspace(0, self.D/2, integrand.shape[1]))

        power_far = np.abs(E_far)**2
        radiation_pattern_dB = 10 * np.log10(power_far / np.max(power_far))

        indices_3dB = np.where(radiation_pattern_dB >= -3)[0]
        hpbw = theta_deg[indices_3dB[-1]] - theta_deg[indices_3dB[0]] if len(indices_3dB) > 0 else 0.0
        
        area = np.pi * (self.D / 2)**2
        eficiencia = 0.60 if delta_x == 0 else 0.60 * (1 - 0.12 * np.abs(delta_x)/self.D)
        ganho_dBi = 10 * np.log10((4 * np.pi * area / self.wavelength**2) * eficiencia)

        return {
            "X": X, "Y": Y, "Z": Z, "f": self.f, "theta_deg": theta_deg,
            "pattern_dB": radiation_pattern_dB, "hpbw": hpbw, "ganho": ganho_dBi
        }

    def calcular_campo_focal(self, theta_inc_deg=0.0):
        """
        Calcula a distribuição de energia no plano focal (Z = f)
        quando uma onda chega com um ângulo theta_inc.
        """
        theta_inc = np.deg2rad(theta_inc_deg)
        
        # Malha do plano focal (região pequena ao redor do foco geométrico)
        xf = np.linspace(-0.2, 0.2, 40)
        yf = np.linspace(-0.2, 0.2, 40)
        XF, YF = np.meshgrid(xf, yf)
        
        # Discretização do refletor para integração
        rho = np.linspace(0, self.D / 2, 40)
        phi = np.linspace(0, 2 * np.pi, 40)
        Rho, Phi = np.meshgrid(rho, phi)
        X = Rho * np.cos(Phi)
        Y = Rho * np.sin(Phi)
        Z = (X**2 + Y**2) / (4 * self.f)
        
        # Campo induzido no prato pela onda vinda do espaço (fase linear)
        E_prato = np.exp(1j * self.k * X * np.sin(theta_inc))
        
        # Campo resultante em cada ponto do plano focal (Integral de Kirchhoff simplificada)
        E_focal = np.zeros_like(XF, dtype=complex)
        
        for r in range(XF.shape[0]):
            for c in range(XF.shape[1]):
                # Distância de cada ponto do prato até o ponto (r, c) do plano focal
                R = np.sqrt((X - XF[r, c])**2 + (Y - YF[r, c])**2 + (self.f - Z)**2)
                integrand = (E_prato / R) * np.exp(-1j * self.k * R) * Rho
                
                int_phi = simpson(integrand, x=np.linspace(0, 2*np.pi, integrand.shape[0]), axis=0)
                E_focal[r, c] = simpson(int_phi, x=np.linspace(0, self.D/2, integrand.shape[1]))
                
        intensidade_focal = np.abs(E_focal)**2
        return XF, YF, intensidade_focal / np.max(intensidade_focal)
    def simular_passagem_satelite(self, passo_tempo):
        """
        Simula a trajetória de um satélite LEO cruzando o céu.
        passo_tempo varia de 0 a 100 (início ao fim da passagem).
        """
        # Mapeia o passo de tempo para um ângulo de órbita de -12 a +12 graus
        angulo_satelite = -12.0 + (passo_tempo / 100.0) * 24.0
        
        # O deslocamento ideal no plano focal para capturar esse ângulo
        # Relação aproximada: delta_x_otimo = -F * sin(theta)
        delta_x_otimo = -self.f * np.sin(np.deg2rad(angulo_satelite))
        
        # Link Budget simplificado: Perda de sinal aumenta nas bordas da passagem
        perda_varredura = 10 * np.log10(np.cos(np.deg2rad(angulo_satelite)) ** 4)
        
        return angulo_satelite, delta_x_otimo, perda_varredura