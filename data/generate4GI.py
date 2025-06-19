import numpy as np
from scipy.integrate import odeint
import pandas as pd
import matplotlib.pyplot as plt

class FourGIModel:
    def __init__(self, patient_type='T2DM'):
        """
        Initialize 4GI model
        patient_type: 'T2DM' or 'HV' (Healthy Volunteers)
        """
        self.patient_type = patient_type
        self._set_parameters()
        self._set_baseline_values()
        
    def _set_parameters(self):
        """Set model parameters (based on paper Table 3 and Table 4)"""
        # Glucose parameters
        if self.patient_type == 'T2DM':
            self.CLglc = 1.72  # Glucose clearance (L/h)
            self.CLglci = 0.0256  # Insulin-dependent glucose clearance (L/h/pmol)
        else:  # HV
            self.CLglc = 5.36
            self.CLglci = 0.072
            
        self.Qglc = 26.5  # Inter-compartmental glucose clearance (L/h)
        self.VCglc = 9.33  # Central glucose distribution volume (L)
        self.VPglc = 8.56  # Peripheral glucose distribution volume (L)
        
        # Insulin parameters
        self.CLins = 73.2  # Insulin clearance (L/h)
        self.VCins = 6.09  # Insulin distribution volume (L)
        self.Ke0ins = np.exp(-0.159)  # Effect compartment rate constant
        
        # GLP-1 parameters
        self.VCglp = 16.0  # GLP-1 distribution volume (L)
        self.VM_GLP = np.exp(7.97)  # Maximum clearance rate
        self.KM_GLP = np.exp(4.91)  # Michaelis constant
        
        # Glucagon parameters
        self.CLglg = 453.2  # Glucagon clearance (L/h)
        self.VCglg = 64.6  # Glucagon distribution volume (L)
        
        # GIP parameters
        self.CLgip = 86.8  # GIP clearance (L/h)
        self.VCgip = 9.21  # GIP distribution volume (L)
        self.Qgip = 49.4  # GIP inter-compartmental clearance
        self.VPgip = 22.8  # GIP peripheral distribution volume
        
        # Effect parameters
        self.GLCINS_S = 2.46  # Glucose-stimulated insulin secretion
        self.EMAX_1 = np.exp(2.37)  # Maximum effect of GLP-1 on insulin secretion
        self.EC50_1 = np.exp(3.29)  # EC50 of GLP-1
        self.HILL_1 = 1.79  # Hill coefficient
        
        self.EMAX_4 = 6.73  # Maximum effect of glucagon on glucose production
        self.EC50_4 = np.exp(4.59)  # EC50 of glucagon
        
        # Food effect parameters
        self.FDGLP = 0.0102  # Food effect on GLP-1
        self.FDGIP = 0.0343  # Food effect on GIP
        self.FDGLG = 0.00329  # Food effect on glucagon
        
    def _set_baseline_values(self):
        """Set baseline values"""
        self.BSLglc = 7.0  # Glucose baseline (mmol/L)
        self.BSLins = 50.0  # Insulin baseline (pmol/L)
        self.BSLglp = 10.0  # GLP-1 baseline (pmol/L)
        self.BSLglg = 25.0  # Glucagon baseline (pmol/L)
        self.BSLgip = 20.0  # GIP baseline (pmol/L)
        
    def model_equations(self, y, t, meal_input=0):
        """
        4GI model differential equations
        y: state variables [glucose_central, insulin, glp1, glucagon, gip, glucose_peripheral, 
                   insulin_effect, gip_peripheral]
        t: time
        meal_input: meal input (mmol glucose)
        """
        # Unpack state variables
        Gc, Ins, GLP, Glg, GIP, Gp, InsE, GIPp = y
        
        # Calculate concentrations
        Cglc = Gc / self.VCglc
        Cins = Ins / self.VCins
        Cglp = GLP / self.VCglp
        Cglg = Glg / self.VCglg
        Cgip = GIP / self.VCgip
        
        # Effect calculations
        # Effect of GLP-1 on insulin secretion
        GLPINS_S = self.EMAX_1 * (Cglp/self.EC50_1)**self.HILL_1 / \
                   (1 + (Cglp/self.EC50_1)**self.HILL_1)
        GLPINS_S0 = self.EMAX_1 * (self.BSLglp/self.EC50_1)**self.HILL_1 / \
                    (1 + (self.BSLglp/self.EC50_1)**self.HILL_1)
        
        # Effect of glucagon on glucose production
        GLGGLC_S = self.EMAX_4 * (Cglg/self.EC50_4) / (1 + Cglg/self.EC50_4)
        GLGGLC_S0 = self.EMAX_4 * (self.BSLglg/self.EC50_4) / (1 + self.BSLglg/self.EC50_4)
        glgEFFglc = (1 + GLGGLC_S) / (1 + GLGGLC_S0)
        
        # Glucose feedback on glucagon
        if self.patient_type == 'T2DM':
            POW_2 = 0.925 if Cglc >= self.BSLglc else 0
        else:
            POW_2 = 0.925 if Cglc >= self.BSLglc else 0.327
        glcEFFglg = (self.BSLglc / Cglc) ** POW_2 if Cglc > 0 else 1
        
        # Baseline production rates
        KINglc = self.BSLglc * (self.CLglc + self.CLglci * self.BSLins)
        KINins = self.BSLins * self.CLins / (1 + GLPINS_S0 * self.BSLglc**self.GLCINS_S)
        KINglp = self.VM_GLP * self.BSLglp * self.VCglp / (self.KM_GLP + self.BSLglp)
        KINglg = self.BSLglg * self.CLglg
        KINgip = self.BSLgip * self.CLgip
        
        # Food effects
        meal_effect = meal_input * 10  # Amplify food effects to produce noticeable response
        FDGLP_S = self.FDGLP * meal_effect if meal_effect > 0 else 0
        FDGIP_S = self.FDGIP * meal_effect if meal_effect > 0 else 0
        FDGLG_S = self.FDGLG * meal_effect if meal_effect > 0 else 0
        
        # Differential equations
        # Glucose kinetics
        K27 = self.Qglc / self.VCglc
        K72 = self.Qglc / self.VPglc
        dGc_dt = meal_input + KINglc * glgEFFglc - K27 * Gc + K72 * Gp - \
                 (self.CLglc/self.VCglc) * Gc - (self.CLglci * InsE / self.VCglc) * Gc
        
        # Insulin kinetics
        STglc = GLPINS_S  # Simplified: only consider GLP-1 effect
        dIns_dt = KINins * (1 + STglc * Cglc**self.GLCINS_S) - \
                  (self.CLins/self.VCins) * Ins
        
        # GLP-1 kinetics
        dGLP_dt = KINglp * (1 + FDGLP_S) - \
                  self.VM_GLP * Cglp / (self.KM_GLP + Cglp)
        
        # Glucagon kinetics
        dGlg_dt = KINglg * (1 + FDGLG_S) * glcEFFglg - \
                  (self.CLglg/self.VCglg) * Glg
        
        # GIP kinetics
        K612 = self.Qgip / self.VCgip
        K126 = self.Qgip / self.VPgip
        dGIP_dt = KINgip * (1 + FDGIP_S) - (self.CLgip/self.VCgip) * GIP - \
                  K612 * GIP + K126 * GIPp
        
        # Peripheral glucose
        dGp_dt = K27 * Gc - K72 * Gp
        
        # Insulin effect compartment
        dInsE_dt = self.Ke0ins * (Cins - InsE)
        
        # Peripheral GIP
        dGIPp_dt = K612 * GIP - K126 * GIPp
        
        return [dGc_dt, dIns_dt, dGLP_dt, dGlg_dt, dGIP_dt, dGp_dt, dInsE_dt, dGIPp_dt]
    
    def simulate(self, duration_hours=5, sampling_interval_min=5, meal_times=[], meal_sizes=[]):
        """
        Simulate 4GI model
        duration_hours: simulation duration (hours)
        sampling_interval_min: sampling interval (minutes)
        meal_times: list of meal times (hours)
        meal_sizes: list of meal sizes (glucose mmol)
        """
        # Time points
        t_minutes = np.arange(0, duration_hours * 60 + sampling_interval_min, sampling_interval_min)
        t_hours = t_minutes / 60
        
        # Initial conditions
        y0 = [
            self.BSLglc * self.VCglc,  # Glucose central compartment
            self.BSLins * self.VCins,  # Insulin
            self.BSLglp * self.VCglp,  # GLP-1
            self.BSLglg * self.VCglg,  # Glucagon
            self.BSLgip * self.VCgip,  # GIP
            self.BSLglc * self.VPglc,  # Glucose peripheral compartment
            self.BSLins,               # Insulin effect
            self.BSLgip * self.VPgip  # GIP peripheral compartment
        ]
        
        # Numerical integration
        solution = []
        current_y = y0
        
        for i in range(len(t_hours) - 1):
            t_span = [t_hours[i], t_hours[i+1]]
            
            # Check for meal input
            meal_input = 0
            for meal_time, meal_size in zip(meal_times, meal_sizes):
                if t_hours[i] <= meal_time < t_hours[i+1]:
                    meal_input = meal_size / (t_span[1] - t_span[0])  # Distribute over time interval
            
            # Integrate one time step
            sol = odeint(self.model_equations, current_y, t_span, args=(meal_input,))
            solution.append(sol[0])
            current_y = sol[-1]
        
        solution.append(current_y)
        solution = np.array(solution)
        
        # Convert to concentrations
        glucose = solution[:, 0] / self.VCglc
        insulin = solution[:, 1] / self.VCins
        glp1 = solution[:, 2] / self.VCglp
        glucagon = solution[:, 3] / self.VCglg
        gip = solution[:, 4] / self.VCgip
        
        return t_hours, glucose, insulin, glp1, glucagon, gip
    
    def add_measurement_noise(self, data, cv=0.1):
        """
        Add measurement noise
        cv: coefficient of variation (relative standard deviation)
        """
        noise = np.random.normal(0, cv * np.abs(data), size=data.shape)
        return data + noise
    
    def generate_dataset(self, duration_hours=5, sampling_interval_min=5, 
                        meal_times=[1, 3], meal_sizes=[75, 50], 
                        noise_cv=0.1, n_subjects=10):
        """
        Generate 4GI dataset
        """
        datasets = []
        
        for subject_id in range(n_subjects):
            # Add some individual variability for each subject
            self.BSLglc *= np.random.normal(1, 0.1)
            self.BSLins *= np.random.normal(1, 0.15)
            self.BSLglp *= np.random.normal(1, 0.15)
            self.BSLglg *= np.random.normal(1, 0.15)
            self.BSLgip *= np.random.normal(1, 0.15)
            
            # Simulation
            t, glucose, insulin, glp1, glucagon, gip = self.simulate(
                duration_hours, sampling_interval_min, meal_times, meal_sizes
            )
            
            # Add noise
            glucose_noisy = self.add_measurement_noise(glucose, noise_cv)
            insulin_noisy = self.add_measurement_noise(insulin, noise_cv * 1.5)
            glp1_noisy = self.add_measurement_noise(glp1, noise_cv * 1.5)
            glucagon_noisy = self.add_measurement_noise(glucagon, noise_cv * 1.2)
            gip_noisy = self.add_measurement_noise(gip, noise_cv * 1.3)
            
            # Create dataframe
            df = pd.DataFrame({
                'subject_id': subject_id,
                'time_hours': t,
                'time_minutes': t * 60,
                'glucose_mmol_L': glucose_noisy,
                'insulin_pmol_L': insulin_noisy,
                'glp1_pmol_L': glp1_noisy,
                'glucagon_pmol_L': glucagon_noisy,
                'gip_pmol_L': gip_noisy,
                'meal_indicator': [1 if any(abs(t[i] - mt) < 0.01 for mt in meal_times) else 0 
                                 for i in range(len(t))]
            })
            
            datasets.append(df)
            
            # Reset baseline values
            self._set_baseline_values()
        
        return pd.concat(datasets, ignore_index=True)

# Usage example
if __name__ == "__main__":
    # Create model instance
    model = FourGIModel(patient_type='T2DM')
    
    # Generate dataset
    dataset = model.generate_dataset(
        duration_hours=5,
        sampling_interval_min=5,
        meal_times=[0.5, 2.5],  # Meals at 30 minutes and 2.5 hours
        meal_sizes=[75, 50],    # Glucose amount (mmol)
        noise_cv=0.1,           # 10% measurement noise
        n_subjects=10           # 10 subjects
    )
    
    # Save dataset
    dataset.to_csv('4gi_dataset.csv', index=False)
    
    # Visualize data for the first subject
    subject_0_data = dataset[dataset['subject_id'] == 0]
    
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    
    biomarkers = ['glucose_mmol_L', 'insulin_pmol_L', 'glp1_pmol_L', 
                  'glucagon_pmol_L', 'gip_pmol_L']
    titles = ['Glucose (mmol/L)', 'Insulin (pmol/L)', 'GLP-1 (pmol/L)', 
              'Glucagon (pmol/L)', 'GIP (pmol/L)']
    
    for ax, biomarker, title in zip(axes, biomarkers, titles):
        ax.plot(subject_0_data['time_hours'], subject_0_data[biomarker], 'b-', linewidth=2)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        
        # Mark meal times
        meal_times = subject_0_data[subject_0_data['meal_indicator'] == 1]['time_hours']
        for mt in meal_times:
            ax.axvline(x=mt, color='r', linestyle='--', alpha=0.5)
    
    axes[-1].set_xlabel('Time (hours)')
    plt.suptitle('4GI Model Simulation - Subject 0', fontsize=16)
    plt.tight_layout()
    plt.savefig('4gi_simulation_example.png', dpi=300)
    plt.show()
    
    print(f"Dataset generated and saved to '4gi_dataset.csv'")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Contains {dataset['subject_id'].nunique()} subjects")
    print(f"Each subject has {len(dataset[dataset['subject_id'] == 0])} time points")
    print("\nFirst 5 rows of dataset:")
    print(dataset.head())