    def update_C(self,X,prev_C,u,v):
        C=prev_C+torch.dot(v,v.t())*u@u.t()
        M=u@v.t()@X
        C=C-M-M.t()
        return C
