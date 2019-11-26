'''
**********************************************
Input must be a pytorch tensor
**********************************************
'''

def quantize(x,input_compress_settings={}):
    compress_settings={'n':6}
    compress_settings.update(input_compress_settings)
    #assume that x is a torch tensor
    
    n=compress_settings['n']
    #print('n:{}'.format(n))
    x=x.float()
    x_norm=torch.norm(x,p=float('inf'))
    
    sgn_x=((x>0).float()-0.5)*2
    
    p=torch.div(torch.abs(x),x_norm)
    renormalize_p=torch.mul(p,n)
    floor_p=torch.floor(renormalize_p)
    compare=torch.rand_like(floor_p)
    final_p=renormalize_p-floor_p
    margin=(compare < final_p).float()
    xi=(floor_p+margin)/n
    
    
    
    Tilde_x=x_norm*sgn_x*xi
    
    return Tilde_x




def sparse_randomized(x,input_compress_settings={}):
    max_iteration=10000
    compress_settings={'p':0.8}
    compress_settings.update(input_compress_settings)
    #p=compress_settings['p']
    #vec_x=x.flatten()
    #out=torch.dropout(vec_x,1-p,train=True)
    #out=out/p
    vec_x=x.flatten()
    d = int(len(vec_x))
    p=compress_settings['p']
    
    abs_x=torch.abs(vec_x)
    #d=torch.prod(torch.Tensor(x.size()))
    out=torch.min(p*d*abs_x/torch.sum(abs_x),torch.ones_like(abs_x))
    i=0
    while True:
        i+=1
        #print(i)
        if i>=max_iteration:
            raise ValueError('Too much operations!')
        temp=out.detach()
            
        cI=1-torch.eq(out,1).float()
        c=(p*d-d+torch.sum(cI))/torch.sum(out*cI)
        if c<=1:
            break
        out=torch.min(c*out,torch.ones_like(out))
        if torch.sum(1-torch.eq(out,temp)):
            break
    
    z=torch.rand_like(out)
    out=vec_x*(z<out).float()/out

    out=out.reshape(x.shape)

    #out=out.reshape(x.shape)
    return out
