def update_conv_layer(self, dL_pool, input_of_layer, kernels, conv_outputL, pool_indices):
        """
        Funçao para calcular o gradiente dos kernels de uma layer
        
        Parametros
        -dL_pool: Gradiente do Pool step que é recebido da layer anterior
        -input_of_layer: Input que aquela layer de kernels recebe, que é o resultado do pool
        step da layer anterior
        -kernels: kernels daquela layer
        -conv_outputL: Resultados da convoluçao daquela layer antes da ReLU
        -pool_indices: indices dos valores escolhidos pelo Pool step

        Returns
        -dL_kernels: Gradiente para ajustar os kernels
        -dLa_pool: Gradiente a ser passado para a proxima layer
        """
        dL_kernels = np.zeros_like(kernels)
        dLa_pool = [np.zeros_like(pool) for pool in input_of_layer]

        feature_idx = 0
        for la_feature_idx in range(len(input_of_layer)):
            for k in range(kernels.shape[0]):
                # Get stored values
                la_feature = input_of_layer[la_feature_idx]

                 # Gradient for this feature
                dFeature = dL_pool[feature_idx] #5x5
                
                # Backprop through pooling layer
                dPool = np.zeros_like(conv_outputL) #11x11
                # Set gradient only at the max position
                i = 0
                for l in range(dFeature.shape[0]):
                    j = 0
                    for c in range(dFeature.shape[1]):
                        max_i, max_j = pool_indices[l][c]
                        dPool[i:i+2, j:j+2][max_i - (max_i//2)*2, max_j - (max_j//2)*2] = dFeature[l][c]

                        j +=2
                    i +=2

                # Backprop through ReLU
                dReLU = dPool * self.ReLu_derivative(conv_outputL) #11x11

                
                # Backprop through convolution
                # Update kernel gradients
                for i in range(la_feature.shape[0] - 2):
                    for j in range(la_feature.shape[1] - 2):
                        if dReLU[i, j] != 0:
                            dL_kernels[k] += dReLU[i, j] * la_feature[i:i+3, j:j+3]
                            # Also accumulate gradients for previous layer
                            dLa_pool[la_feature_idx][i:i+3, j:j+3] += dReLU[i, j] * kernels[k]
                
                feature_idx += 1
        
        return dL_kernels, dLa_pool
