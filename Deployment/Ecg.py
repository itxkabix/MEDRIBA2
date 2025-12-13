# Ecg.py
# Robust ECG image -> 1D signal pipeline with diagnostics and improved fallbacks
# Replace existing Ecg.py with this file.

from skimage.io import imread
from skimage import color
from skimage.filters import threshold_otsu, gaussian
from skimage.transform import resize
from skimage import measure, feature
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from natsort import natsorted
import warnings
import hashlib
import traceback

warnings.filterwarnings("ignore")


class ECG:
    def __init__(self, verbose=True, debug=False, n_points=255):
        """
        verbose: print logs
        debug: write extra debug CSVs and prints
        n_points: output length per lead
        """
        self.verbose = verbose
        self.debug = debug
        self.n_points = n_points

    def _log(self, *args):
        if self.verbose:
            print(*args)

    def _debug(self, *args):
        if self.debug:
            print(*args)

    # ---- utility: cleanup stale files ----
    def clear_stale_files(self, work_dir="."):
        """
        Remove previously produced Scaled_1DLead_* and Combined_1D_AllLeads.csv
        Useful to ensure previous runs don't contaminate current run.
        """
        for i in range(1, 13):
            f = os.path.join(work_dir, f"Scaled_1DLead_{i}.csv")
            if os.path.isfile(f):
                try:
                    os.remove(f)
                    self._log(f"Removed stale file: {f}")
                except Exception:
                    self._log("Could not remove", f)
        combo = os.path.join(work_dir, "Combined_1D_AllLeads.csv")
        if os.path.isfile(combo):
            try:
                os.remove(combo)
                self._log(f"Removed stale file: {combo}")
            except Exception:
                self._log("Could not remove", combo)

    # ---- input image read ----
    def getImage(self, image):
        """
        Accepts either a file path or a file-like object (as in Streamlit upload)
        Returns an image in numpy array form (RGB float [0..1]).
        """
        try:
            img = imread(image)
            # If grayscale convert to RGB-like 3-channel
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            img = img_as_float(img)
            return img
        except Exception as e:
            raise ValueError(f"Could not read image: {e}")

    # ---- grayscale + resize ----
    def GrayImgae(self, image, out_shape=(1572, 2213)):
        """
        Convert to grayscale and resize to a working shape (default approximately original pipeline).
        """
        try:
            image_gray = color.rgb2gray(image)
            image_gray = resize(image_gray, out_shape, preserve_range=True, anti_aliasing=True)
            return image_gray
        except Exception as e:
            raise RuntimeError(f"GrayImgae failed: {e}")

    # ---- dividing leads with scaling support ----
    def DividingLeads(self, image):
        """
        Divides a typical 12-lead ECG page into 13 Leads including long lead.
        Uses proportional coordinates so images with different resolutions are handled.
        Returns list of 13 lead images (12 + long lead).
        """
        try:
            h, w = image.shape[0], image.shape[1]
            self._log("Input image shape (H x W):", h, "x", w)

            # Reference coordinates from original pipeline, scale them to actual image size
            ref_h, ref_w = 1480, 2125
            scale_h = h / ref_h
            scale_w = w / ref_w

            def S(y0, y1, x0, x1):
                return image[
                    int(y0 * scale_h):int(y1 * scale_h),
                    int(x0 * scale_w):int(x1 * scale_w)
                ]

            Lead_1 = S(300, 600, 150, 643)   # Lead 1
            Lead_2 = S(300, 600, 646, 1135)  # Lead aVR
            Lead_3 = S(300, 600, 1140, 1625)  # Lead V1
            Lead_4 = S(300, 600, 1630, 2125)  # Lead V4
            Lead_5 = S(600, 900, 150, 643)   # Lead 2
            Lead_6 = S(600, 900, 646, 1135)  # Lead aVL
            Lead_7 = S(600, 900, 1140, 1625) # Lead V2
            Lead_8 = S(600, 900, 1630, 2125) # Lead V5
            Lead_9 = S(900, 1200, 150, 643)  # Lead 3
            Lead_10 = S(900, 1200, 646, 1135) # Lead aVF
            Lead_11 = S(900, 1200, 1140, 1625) # Lead V3
            Lead_12 = S(900, 1200, 1630, 2125) # Lead V6
            Lead_13 = S(1250, 1480, 150, 2125) # Long Lead

            Leads = [Lead_1, Lead_2, Lead_3, Lead_4,
                     Lead_5, Lead_6, Lead_7, Lead_8,
                     Lead_9, Lead_10, Lead_11, Lead_12, Lead_13]

            # Save 12-lead grid (guarded)
            try:
                fig, ax = plt.subplots(4, 3, figsize=(10, 10))
                x_counter = 0
                y_counter = 0
                for x, y in enumerate(Leads[:12]):
                    ax[x_counter][y_counter].imshow(y)
                    ax[x_counter][y_counter].axis('off')
                    ax[x_counter][y_counter].set_title(f"Lead {x+1}")
                    if (x+1) % 3 == 0:
                        x_counter += 1
                        y_counter = 0
                    else:
                        y_counter += 1
                fig.tight_layout()
                fig.savefig('Leads_1-12_figure.png')
                plt.close(fig)
            except Exception as e:
                self._log("Warning: could not save Leads_1-12_figure:", e)

            # Save long lead view
            try:
                fig1, ax1 = plt.subplots(figsize=(8, 4))
                ax1.imshow(Lead_13)
                ax1.axis('off')
                ax1.set_title("Long Lead 13")
                fig1.tight_layout()
                fig1.savefig('Long_Lead_13_figure.png')
                plt.close(fig1)
            except Exception as e:
                self._log("Warning: could not save Long_Lead_13_figure:", e)

            return Leads
        except Exception as e:
            raise RuntimeError(f"DividingLeads failed: {e}")

    # ---- preprocessing leads (thresholding, resizing) ----
    def PreprocessingLeads(self, Leads):
        """
        Convert each lead to binary image using gaussian + Otsu threshold,
        with per-lead fallback to zeros. Returns list of processed binary arrays.
        """
        processed = []
        try:
            fig2, ax2 = plt.subplots(4, 3, figsize=(10, 10))
            x_counter = 0
            y_counter = 0
            for x, y in enumerate(Leads[:12]):
                try:
                    grayscale = color.rgb2gray(y)
                    blurred_image = gaussian(grayscale, sigma=1)
                    global_thresh = threshold_otsu(blurred_image)
                    binary_global = blurred_image < global_thresh
                    binary_global = resize(binary_global, (300, 450), preserve_range=True, anti_aliasing=True)
                except Exception as e:
                    self._log(f"Warning: preprocessing lead {x+1} failed, using zeros. Err: {e}")
                    binary_global = np.zeros((300, 450))

                processed.append(binary_global)
                try:
                    ax2[x_counter][y_counter].imshow(binary_global, cmap="gray")
                    ax2[x_counter][y_counter].axis('off')
                    ax2[x_counter][y_counter].set_title(f"pre-processed Lead {x+1}")
                except Exception:
                    pass
                if (x+1) % 3 == 0:
                    x_counter += 1
                    y_counter = 0
                else:
                    y_counter += 1

            fig2.tight_layout()
            fig2.savefig('Preprossed_Leads_1-12_figure.png')
            plt.close(fig2)

            # Lead 13
            try:
                grayscale = color.rgb2gray(Leads[-1])
                blurred_image = gaussian(grayscale, sigma=1)
                global_thresh = threshold_otsu(blurred_image)
                binary_global = blurred_image < global_thresh
                binary_global = resize(binary_global, (300, 200), preserve_range=True, anti_aliasing=True)
            except Exception as e:
                self._log(f"Warning: preprocessing long lead failed, using zeros. Err: {e}")
                binary_global = np.zeros((300, 200))

            processed.append(binary_global)
            try:
                fig3, ax3 = plt.subplots(figsize=(8, 4))
                ax3.imshow(binary_global, cmap='gray')
                ax3.axis('off')
                ax3.set_title("Lead 13 Preprocessed")
                fig3.tight_layout()
                fig3.savefig('Preprossed_Leads_13_figure.png')
                plt.close(fig3)
            except Exception as e:
                self._log("Warning: could not save Preprossed_Leads_13_figure:", e)

            return processed
        except Exception as e:
            raise RuntimeError(f"PreprocessingLeads failed: {e}")

    # ---- helper: robust contour extraction with fallbacks ----
    def _extract_contour_from_binary(self, binary_img):
        """
        Try multiple strategies to extract a single contour representing waveform:
        1) find_contours on binary image
        2) find_contours on Canny edges
        3) column-wise argmin on blurred grayscale (fallback)
        Returns an (M,2) array of points (row, col) or raises.
        """
        # First attempt: measure.find_contours on binary (expected)
        contours = measure.find_contours(binary_img, 0.8)
        if contours:
            # pick the longest contour (most points)
            contour = max(contours, key=lambda c: c.shape[0])
            if contour.ndim == 2 and contour.shape[1] >= 2:
                return contour

        # Second attempt: use Canny edges
        try:
            edges = feature.canny(binary_img.astype(float))
            contours2 = measure.find_contours(edges, 0.5)
            if contours2:
                contour = max(contours2, key=lambda c: c.shape[0])
                if contour.ndim == 2 and contour.shape[1] >= 2:
                    return contour
        except Exception:
            pass

        # Third attempt: column-wise argmin on original grayscale (assume waveform darker)
        try:
            # For each column, find the row position of min value in binary_img (or argmin on smoothed)
            # Convert to float image if needed
            if binary_img.ndim != 2:
                raise ValueError("binary_img is not 2D")
            h, w = binary_img.shape
            cols = []
            for col in range(w):
                col_vals = binary_img[:, col]
                # use argmin if the signal is darker; otherwise argmax
                idx = int(np.argmin(col_vals))
                cols.append([idx, col])
            contour = np.array(cols, dtype=float)
            return contour
        except Exception as e:
            raise RuntimeError(f"All contour strategies failed: {e}")

    # ---- signal extraction + scaling ----
    def SignalExtraction_Scaling(self, Leads):
        """
        For each of the 12 primary leads:
          - Convert to gray, blur, threshold
          - Find contour / waveform (with fallback strategies)
          - Resample/interpolate to self.n_points
          - Scale to 0..1 using MinMaxScaler
          - Save each lead as 'Scaled_1DLead_{n}.csv' containing a single row of length n_points (header=False)
        """
        n_points = self.n_points
        try:
            fig4, ax4 = plt.subplots(4, 3, figsize=(10, 10))
            x_counter = 0
            y_counter = 0

            for idx, y in enumerate(Leads[:12]):
                orig_len = None
                test = None
                try:
                    grayscale = color.rgb2gray(y)
                    blurred_image = gaussian(grayscale, sigma=0.7)
                    global_thresh = threshold_otsu(blurred_image)
                    binary_global = blurred_image < global_thresh
                    binary_global = resize(binary_global, (300, 450), preserve_range=True, anti_aliasing=True)

                    # try extracting a contour with fallbacks
                    contour = self._extract_contour_from_binary(binary_global)
                    orig_len = contour.shape[0]

                    # ensure valid shape
                    if contour.ndim != 2 or contour.shape[1] < 2:
                        raise ValueError("Invalid contour shape")

                    # interpolate to n_points along the contour's index
                    if orig_len != n_points:
                        new_idx = np.linspace(0, orig_len - 1, n_points)
                        old_idx = np.arange(orig_len)
                        test0 = np.interp(new_idx, old_idx, contour[:, 0])
                        test1 = np.interp(new_idx, old_idx, contour[:, 1])
                        test = np.vstack([test0, test1]).T
                    else:
                        test = contour.copy()
                except Exception as e:
                    # fallback: create a reasonable baseline trace using column-wise median positions (better than flat zero)
                    self._log(f"Warning: contour extraction failed for lead {idx+1}: {e}. Using column-wise fallback.")
                    try:
                        # Attempt fallback from grayscale to extract a vertical position per column
                        grayscale = color.rgb2gray(Leads[idx])
                        small = resize(grayscale, (60, 200), preserve_range=True, anti_aliasing=True)
                        col_positions = []
                        for col in range(small.shape[1]):
                            col_vals = small[:, col]
                            pos = float(np.argmin(col_vals))  # position of min (waveform darker)
                            col_positions.append(pos)
                        # map positions to 0..n_points and pair with column index (scaled)
                        cols = np.linspace(0, small.shape[1] - 1, n_points)
                        rows = np.interp(np.linspace(0, small.shape[1] - 1, n_points), np.arange(len(col_positions)), col_positions)
                        test = np.vstack([rows, cols]).T
                    except Exception as e2:
                        # Last-ditch: linear ramp as a fallback (rare)
                        self._log(f"Fallback 2 failed for lead {idx+1}: {e2}. Using flat ramp.")
                        test = np.column_stack([np.linspace(0, 1, n_points), np.linspace(0, 1, n_points)])

                # plot (invert y to match original)
                try:
                    r = idx // 3
                    c = idx % 3
                    ax4[r][c].invert_yaxis()
                    ax4[r][c].plot(test[:, 1], test[:, 0], linewidth=1)
                    ax4[r][c].axis('image')
                    ax4[r][c].set_title(f"Contour Lead {idx+1}")
                except Exception:
                    pass

                # Scale vertical positions to 0..1
                try:
                    sig = np.asarray(test[:, 0], dtype=float).reshape(-1, 1)
                    scaler = MinMaxScaler()
                    scaled = scaler.fit_transform(sig).flatten()
                    # write single-row CSV, header=False
                    row_df = pd.DataFrame([scaled])
                    fname = f"Scaled_1DLead_{idx+1}.csv"
                    row_df.to_csv(fname, index=False, header=False)
                    # debug outputs
                    if self.debug:
                        debug_info = {
                            'lead': idx + 1,
                            'orig_len': orig_len,
                            'test_shape': test.shape,
                            'scaled_min': float(np.min(scaled)),
                            'scaled_max': float(np.max(scaled)),
                            'scaled_mean': float(np.mean(scaled)),
                        }
                        print("DEBUG_LEAD_SIGNAL:", debug_info)
                        pd.DataFrame([scaled]).to_csv(f"DEBUG_Scaled_1DLead_{idx+1}.csv", index=False, header=False)
                except Exception as e:
                    self._log(f"Warning: scaling/saving failed for lead {idx+1}: {e}")
                    # Save zeros to keep file consistent
                    pd.DataFrame([np.zeros(n_points)]).to_csv(f"Scaled_1DLead_{idx+1}.csv", index=False, header=False)

            fig4.tight_layout()
            fig4.savefig('Contour_Leads_1-12_figure.png')
            plt.close(fig4)
            return True
        except Exception as e:
            raise RuntimeError(f"SignalExtraction_Scaling failed: {e}")

    # ---- Combine 1D signals into model input ----
    def CombineConvert1Dsignal(self, expected_leads=12):
        """
        Reads Scaled_1DLead_{1..expected_leads}.csv in numeric order,
        concatenates them horizontally, and returns a single-row DataFrame suitable for model input.
        """
        try:
            files = []
            for i in range(1, expected_leads + 1):
                fname = f"Scaled_1DLead_{i}.csv"
                if not os.path.isfile(fname):
                    raise FileNotFoundError(f"Required lead CSV missing: {fname}")
                files.append(fname)

            # read in order and horizontally concatenate (each CSV is a single row vector)
            dfs = [pd.read_csv(f, header=None) for f in files]

            # Ensure each DF is single-row (flatten if necessary)
            for idx, df in enumerate(dfs):
                if df.shape[0] != 1:
                    # flatten the values to a single row and truncate/pad to n_points if needed
                    vals = df.values.flatten()
                    if vals.size >= self.n_points:
                        vals = vals[: self.n_points]
                    else:
                        pad = np.zeros(self.n_points - vals.size)
                        vals = np.concatenate([vals, pad])
                    dfs[idx] = pd.DataFrame([vals])

            final = pd.concat(dfs, axis=1, ignore_index=True)
            final.columns = range(final.shape[1])
            final.to_csv('Combined_1D_AllLeads.csv', index=False)
            # debug summary
            if self.debug:
                arr = final.values.flatten()
                print("DEBUG_COMBINED shape:", final.shape)
                print("DEBUG_COMBINED first10:", arr[:10].tolist())
                print("DEBUG_COMBINED stats: min/max/mean/std:", float(arr.min()), float(arr.max()), float(arr.mean()), float(arr.std()))
                h = hashlib.md5(np.round(arr, 6).tobytes()).hexdigest()
                print("DEBUG_COMBINED checksum:", h)
            return final
        except Exception as e:
            raise RuntimeError(f"CombineConvert1Dsignal failed: {e}")

    # ---- Dimensionality reduction with PCA + imputation + shape check ----
    def DimensionalReduciton(self, test_final, pca_path='PCA_ECG (1).pkl'):
        """
        Loads PCA from disk and applies transform.
        Handles NaNs by imputing with column means (SimpleImputer).
        If the feature count doesn't match PCA expectation, pads with zeros or truncates.
        Returns transformed dataframe.
        """
        try:
            if not os.path.isfile(pca_path):
                raise FileNotFoundError(f"PCA model not found at {pca_path}")

            pca_loaded_model = joblib.load(pca_path)
            X = test_final.values.astype(float)

            # Impute NaNs using mean
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)

            # PCA expects n_features_in_
            if hasattr(pca_loaded_model, 'n_features_in_'):
                expected = int(pca_loaded_model.n_features_in_)
            else:
                expected = X_imputed.shape[1]

            actual = X_imputed.shape[1]
            if actual != expected:
                self._log(f"DimensionalReduciton: PCA expects {expected} features but input has {actual}. Adjusting...")
                if actual < expected:
                    pad = np.zeros((X_imputed.shape[0], expected - actual))
                    X_adj = np.hstack([X_imputed, pad])
                else:
                    X_adj = X_imputed[:, :expected]
            else:
                X_adj = X_imputed

            # debug PCA input stats
            if self.debug:
                print("DEBUG_PCA input_shape:", X_adj.shape)
                print("DEBUG_PCA input_stats: min,max,mean,std:", float(X_adj.min()), float(X_adj.max()), float(X_adj.mean()), float(X_adj.std()))
                if hasattr(pca_loaded_model, 'n_features_in_'):
                    print("DEBUG_PCA model expects:", pca_loaded_model.n_features_in_)

            result = pca_loaded_model.transform(X_adj)
            final_df = pd.DataFrame(result)
            if self.debug:
                print("DEBUG_PCA output_shape:", final_df.shape)
            return final_df
        except Exception as e:
            raise RuntimeError(f"DimensionalReduciton failed: {e}")

    # ---- Model load + predict (structured return) ----
    def ModelLoad_predict(self, final_df, model_path='Heart_Disease_Prediction_using_ECG (4).pkl', legacy_string=False):
        """
        Loads saved classifier and predicts class. Handles missing model file and returns informative messages.
        legacy_string=False -> returns dict {'message','label','probabilities'}
        legacy_string=True  -> returns plain string message (for backward compatibility)
        """
        try:
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")

            loaded_model = joblib.load(model_path)

            X = final_df.values.astype(float)

            # debug model details
            if self.debug:
                try:
                    print("DEBUG_MODEL type:", type(loaded_model))
                    for attr in ['classes_', 'n_features_in_', 'predict_proba']:
                        print(f"DEBUG_MODEL has {attr}:", hasattr(loaded_model, attr))
                    if hasattr(loaded_model, 'classes_'):
                        print("DEBUG_MODEL classes:", getattr(loaded_model, 'classes_'))
                    print("DEBUG_MODEL input_shape:", X.shape, "input mean/std:", float(X.mean()), float(X.std()))
                except Exception:
                    print("DEBUG_MODEL inspection failed:", traceback.format_exc())

            # If classifier supports predict_proba, return probabilities too
            try:
                probs = loaded_model.predict_proba(X)
            except Exception:
                probs = None

            result = loaded_model.predict(X)
            label = int(result[0]) if hasattr(result, '__len__') else int(result)

            if label == 1:
                msg = "Your ECG corresponds to Myocardial Infarction"
            elif label == 0:
                msg = "Your ECG corresponds to Abnormal Heartbeat"
            elif label == 2:
                msg = "Your ECG is Normal"
            elif label == 3:
                msg = "History of Myocardial Infarction"
            else:
                msg = f"Predicted class: {label}"

            out_dict = {
                'message': msg,
                'label': label,
                'probabilities': probs.tolist() if probs is not None else None
            }

            if legacy_string:
                return msg
            return out_dict
        except Exception as e:
            raise RuntimeError(f"ModelLoad_predict failed: {e}")

    # ---- high-level convenience pipeline ----
    def process_image(self, image_path_or_file, work_dir=".", pca_path='PCA_ECG (1).pkl', model_path='Heart_Disease_Prediction_using_ECG (4).pkl', legacy_string=False):
        """
        High-level helper that runs the full pipeline:
          - clear stale files (in debug mode only)
          - getImage -> GrayImgae -> DividingLeads -> PreprocessingLeads -> SignalExtraction -> Combine -> PCA -> Predict
        Returns the output of ModelLoad_predict.
        """
        try:
            if self.debug:
                self._log("DEBUG MODE: clearing stale files")
                self.clear_stale_files(work_dir)

            img = self.getImage(image_path_or_file)
            gray = self.GrayImgae(img)
            leads = self.DividingLeads(img)
            pre = self.PreprocessingLeads(leads)
            self._log("Running SignalExtraction_Scaling...")
            self.SignalExtraction_Scaling(pre)
            self._log("Combining leads...")
            combined = self.CombineConvert1Dsignal(expected_leads=12)
            self._log("Running PCA...")
            pca_out = self.DimensionalReduciton(combined, pca_path=pca_path)
            self._log("Predicting...")
            pred = self.ModelLoad_predict(pca_out, model_path=model_path, legacy_string=legacy_string)
            return pred
        except Exception as e:
            # Bubble up with trace for easier debugging in Streamlit logs
            tb = traceback.format_exc()
            self._log("Pipeline failed:", e)
            self._log(tb)
            raise

# End of Ecg.py
